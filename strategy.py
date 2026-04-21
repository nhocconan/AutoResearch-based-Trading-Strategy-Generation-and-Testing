#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_DonchianExit
Hypothesis: Daily KAMA trend direction with choppiness regime filter and Donchian(20) breakout for entry.
Works in both bull/bear: KAMA adapts to trend speed, chop filter avoids whipsaws in ranging markets,
Donchian breakout captures momentum after consolidation. Uses 1d timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend context)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d KAMA for adaptive trend ===
    close = prices['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smooth ER with 2 and 30 period EMAs for smoothing constant
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Align KAMA (already 1d, but ensure alignment for safety)
    kama_aligned = kama  # 1d indicator on 1d prices, no alignment needed
    
    # === 1w EMA34 for HTF trend filter ===
    df_1w_close = df_1w['close'].values
    ema_34_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === Choppiness Index (14-period) for regime filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    chop = 100 * np.log10(tr_sum / (hh - ll + 1e-10)) / np.log10(14)
    
    # === Donchian(20) breakout for entry/exit ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) 
            or np.isnan(chop[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        ema_34 = ema_34_1w_aligned[i]
        chop_val = chop[i]
        dc_high = donch_high[i]
        dc_low = donch_low[i]
        
        # Regime: chop < 61.8 = trending (favor trend following), chop > 61.8 = ranging (avoid)
        trending_regime = chop_val < 61.8
        
        if position == 0:
            # Enter long: price > KAMA (trend up) + HTF uptrend + Donchian breakout + trending regime
            long_condition = (price > kama_val) and (price > ema_34) and (price > dc_high[i-1]) and trending_regime
            # Enter short: price < KAMA (trend down) + HTF downtrend + Donchian breakdown + trending regime
            short_condition = (price < kama_val) and (price < ema_34) and (price < dc_low[i-1]) and trending_regime
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price < KAMA (trend reversal) OR Donchian breakdown
            if price < kama_val or price < dc_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA (trend reversal) OR Donchian breakout
            if price > kama_val or price > dc_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0