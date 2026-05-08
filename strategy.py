#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points and volume confirmation.
# Weekly pivot levels (calculated from previous week) act as strong support/resistance.
# Long when price crosses above weekly R1 with volume > 2x average.
# Short when price crosses below weekly S1 with volume > 2x average.
# Exit when price crosses back below/above the weekly pivot point.
# Weekly pivots are calculated once per week and held constant, reducing noise.
# Volume filter ensures only significant breakouts are traded.
# Works in both bull and bear markets as pivots adapt to recent price action.

name = "6h_WeeklyPivot_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    # Using previous week's data to avoid look-ahead
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot points with 1-week lag (use previous week's data)
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe (already lagged by 1 week)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above weekly R1 with volume spike
            if (close[i] > r1_aligned[i] and
                close[i-1] <= r1_aligned[i-1] and  # crossed above this bar
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly S1 with volume spike
            elif (close[i] < s1_aligned[i] and
                  close[i-1] >= s1_aligned[i-1] and  # crossed below this bar
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below weekly pivot
            if close[i] < pivot_aligned[i] and close[i-1] >= pivot_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above weekly pivot
            if close[i] > pivot_aligned[i] and close[i-1] <= pivot_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals