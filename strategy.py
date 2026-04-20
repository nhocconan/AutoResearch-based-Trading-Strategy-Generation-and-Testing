#!/usr/bin/env python3
# 1d_1w_WeeklyPivot_Breakout_Volume
# Hypothesis: Trade weekly pivot point breakouts on daily timeframe with volume confirmation.
# Weekly pivots provide strong support/resistance that work in both bull and bear markets.
# Volume confirms institutional interest. Targets 10-25 trades per year.

name = "1d_1w_WeeklyPivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (using prior week's data)
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    pivot_1w = typical_price_1w
    range_1w = high_1w - low_1w
    
    # Weekly support and resistance levels
    s1_1w = 2 * pivot_1w - high_1w
    r1_1w = 2 * pivot_1w - low_1w
    s2_1w = pivot_1w - range_1w
    r2_1w = pivot_1w + range_1w
    
    # Align weekly levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if (close[i] > r1_aligned[i] * 1.002 and 
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif (close[i] < s1_aligned[i] * 0.998 and 
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 or S2
            if close[i] < s1_aligned[i] * 0.998 or close[i] < s2_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or R2
            if close[i] > r1_aligned[i] * 1.002 or close[i] > r2_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals