#!/usr/bin/env python3
"""
4h_KAMA_Direction_ChopFilter_VolumeSpike
Hypothesis: 4h KAMA trend direction filtered by 1d choppiness regime and volume spike.
Enter long when KAMA turns up (bullish) in low-chop (trending) 1d regime with volume confirmation.
Enter short when KAMA turns down (bearish) in low-chop (trending) 1d regime with volume confirmation.
Exit on ATR(14) trailing stop (2.5*ATR) or opposite KAMA signal.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in bull/bear via 1d regime filter and volume confirmation as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === KAMA (4h) ===
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0)  # 10-period sum of absolute changes
    # Fix array lengths: change is shorter by 10, volatility by 1
    # We'll compute ER for index >= 10
    er = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(10, len(close)):
        if volatility[i] != 0:
            er[i] = change[i-9] / volatility[i]  # change[i-9] corresponds to close[i]-close[i-10]
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=np.float64)
    kama[9] = close[9]  # start at index 9
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d Choppiness Index (CHOP) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no prior close)
    tr_1d[0] = tr1[0]
    # Sum of TR over 14 periods
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(TRAtr) / (HH - LL)) / log10(14)
    chop_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(13, len(close_1d)):
        if hh_1d[i] != ll_1d[i]:
            chop_1d[i] = 100 * np.log10(atr_1d[i] / (hh_1d[i] - ll_1d[i])) / np.log10(14)
        else:
            chop_1d[i] = 0
    # Align to 4h timeframe (use previous completed daily bar)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === Volume spike filter (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(chop_1d_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            # Chop filter: CHOP < 50 indicates trending regime (low chop = trending)
            chop_filter = chop_1d_aligned[i] < 50
            
            # KAMA direction: long when price > KAMA and rising, short when price < KAMA and falling
            kama_rising = kama[i] > kama[i-1]
            kama_falling = kama[i] < kama[i-1]
            
            # Entry logic
            if price > kama[i] and kama_rising and vol_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif price < kama[i] and kama_falling and vol_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: KAMA turns down (price crosses below KAMA)
            elif price < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: KAMA turns up (price crosses above KAMA)
            elif price > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_Direction_ChopFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0