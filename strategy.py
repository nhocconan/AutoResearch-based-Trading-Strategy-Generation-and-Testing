#!/usr/bin/env python3
"""
6h_Pivot_R4_S4_Breakout_With_Volume_Filter
Hypothesis: Use daily pivots (R4/S4) as strong breakout levels. Long when price breaks above R4 with volume > 1.5x average volume; short when price breaks below S4 with volume > 1.5x average volume. Exit on opposite pivot touch (R1/S1) or when volume dries up. R4/S4 represent extreme levels where breakouts indicate strong momentum, reducing false signals. Volume filter ensures breakouts are supported by participation. Targets 12-25 trades/year by requiring rare breakouts with volume confirmation, suitable for 6BTC/ETH in both bull (breakouts continue) and bear (breakdowns continue) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate average volume for filtering (50-period)
    avg_vol = np.full(n, np.nan)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 50:
            vol_sum -= volume[i - 50]
            vol_count -= 1
        if vol_count > 0:
            avg_vol[i] = vol_sum / vol_count
    
    # Get 1D data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily pivots: R4, S4, R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r4 = pivot + 3 * (high_1d - low_1d)
    s4 = pivot - 3 * (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # volume avg + at least one pivot
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(avg_vol[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(r1_6h[i]) or np.isnan(s1_6h[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / avg_vol[i] if avg_vol[i] > 0 else 0
        
        if position == 0:
            # Long entry: price breaks above R4 with volume confirmation
            if close[i] > r4_6h[i] and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S4 with volume confirmation
            elif close[i] < s4_6h[i] and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price touches R1 (take profit) or volume dries up
            if close[i] <= r1_6h[i] or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches S1 (take profit) or volume dries up
            if close[i] >= s1_6h[i] or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R4_S4_Breakout_With_Volume_Filter"
timeframe = "6h"
leverage = 1.0