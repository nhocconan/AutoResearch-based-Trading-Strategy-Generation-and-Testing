#!/usr/bin/env python3
# 6h_1d_WeeklyPivot_Breakout_VolumeTrend
# Hypothesis: Use weekly pivot points from 1w timeframe to determine trend direction, then trade breakouts
# of 1d support/resistance levels on 6h timeframe with volume confirmation. Weekly pivot > price = long bias,
# weekly pivot < price = short bias. This combines multi-timeframe trend filtering with intraday breakout
# logic, working in both bull (breakouts with trend) and bear (mean reversion at extremes) markets.
# Targets 20-40 trades/year by requiring weekly alignment, level proximity, and volume surge.

name = "6h_1d_WeeklyPivot_Breakout_VolumeTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 1d support/resistance levels (pivot-based)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point
    daily_pivot = (high_1d + low_1d + close_1d) / 3.0
    daily_range = high_1d - low_1d
    
    # Key levels: R1, S1, R2, S2
    r1_1d = 2 * daily_pivot - low_1d
    s1_1d = 2 * daily_pivot - high_1d
    r2_1d = daily_pivot + daily_range
    s2_1d = daily_pivot - daily_range
    
    # Align 1d levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long bias: weekly pivot above price -> look for longs at support
            if weekly_pivot_aligned[i] > close[i]:
                # Long near S1 or S2 with volume confirmation
                if (((close[i] <= s1_aligned[i] * 1.002 and close[i] >= s1_aligned[i] * 0.998) or
                     (close[i] <= s2_aligned[i] * 1.002 and close[i] >= s2_aligned[i] * 0.998)) and
                    volume[i] > 1.8 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
            # Short bias: weekly pivot below price -> look for shorts at resistance
            elif weekly_pivot_aligned[i] < close[i]:
                # Short near R1 or R2 with volume confirmation
                if (((close[i] >= r1_aligned[i] * 0.998 and close[i] <= r1_aligned[i] * 1.002) or
                     (close[i] >= r2_aligned[i] * 0.998 and close[i] <= r2_aligned[i] * 1.002)) and
                    volume[i] > 1.8 * volume_ma[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches weekly pivot or shows weakness
            if close[i] >= weekly_pivot_aligned[i] * 0.998 or \
               (close[i] <= s1_aligned[i] * 0.995 and volume[i] < 0.8 * volume_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches weekly pivot or shows weakness
            if close[i] <= weekly_pivot_aligned[i] * 1.002 or \
               (close[i] >= r1_aligned[i] * 1.005 and volume[i] < 0.8 * volume_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals