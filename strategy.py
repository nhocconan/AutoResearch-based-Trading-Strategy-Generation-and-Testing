#!/usr/bin/env python3
"""
12h_1d_Pivot_Bounce_v1
Hypothesis: Use daily pivot levels with volume confirmation on 12h timeframe.
Long when price bounces off S1/S2 with volume > 1.3x average, short when rejected at R1/R2.
Works in bull via bounces from support, in bear via rejections at resistance.
Low trade frequency target: 50-150 total over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_Bounce_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Calculate daily pivot levels (standard floor trader pivots)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    if range_val <= 0:
        return np.zeros(n)
    
    # Daily pivot levels
    r1 = pivot + range_val
    s1 = pivot - range_val
    r2 = pivot + 2 * range_val
    s2 = pivot - 2 * range_val
    
    # Align daily pivot levels to 12h timeframe
    r1_array = np.full(len(df_1d), r1)
    r2_array = np.full(len(df_1d), r2)
    s1_array = np.full(len(df_1d), s1)
    s2_array = np.full(len(df_1d), s2)
    pivot_array = np.full(len(df_1d), pivot)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_array)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_array)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_array)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_array)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_array)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Bounce conditions with volume filter
        long_bounce = ((low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]) or
                       (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i])) and vol_ratio[i] > 1.3
        short_reject = ((high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]) or
                        (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i])) and vol_ratio[i] > 1.3
        
        # Exit conditions: price crosses pivot
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Signal logic
        if long_bounce and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_reject and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals