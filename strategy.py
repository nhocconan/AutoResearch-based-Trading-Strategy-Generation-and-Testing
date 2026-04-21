#!/usr/bin/env python3
"""
1d_KAMA_Regime_Filter_DonchianExit
Hypothesis: Daily timeframe with KAMA trend direction filtered by weekly choppiness regime.
Long when KAMA trending up and market is trending (CHOP < 38.2), short when KAMA trending down and market is trending.
Uses Donchian(20) for exits to capture trend continuation. Designed for low trade frequency (~10-20/year) to minimize fee drag.
Works in both bull/bear via regime filter that avoids ranging markets.
Discrete sizing (0.30) with ATR(14) stoploss (2.5x).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for chop regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Daily KAMA for trend direction ===
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Weekly choppiness regime (CHOP) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    # Chopiness index
    chop = np.where((hh_14 - ll_14) != 0, 
                    100 * np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14), 
                    50)
    # Align weekly chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # === Daily ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Daily Donchian(20) for exits ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) 
            or np.isnan(atr[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        kama_val = kama[i]
        chop_val = chop_aligned[i]
        atr_val = atr[i]
        donch_high = high_20[i]
        donch_low = low_20[i]
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Enter only in trending regime with KAMA direction
            long_condition = is_trending and (price > kama_val)
            short_condition = is_trending and (price < kama_val)
            
            if long_condition:
                signals[i] = 0.30
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.30
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.5x ATR)
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Donchian exit: price breaks below 20-day low
            elif price < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Check stoploss (2.5x ATR)
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Donchian exit: price breaks above 20-day high
            elif price > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_KAMA_Regime_Filter_DonchianExit"
timeframe = "1d"
leverage = 1.0