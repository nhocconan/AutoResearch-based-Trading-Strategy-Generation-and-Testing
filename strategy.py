#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction
Hypothesis: Combine daily Donchian(20) breakouts with weekly pivot direction as trend filter.
In strong weekly trends (price above/below weekly pivot), Donchian breakouts have higher follow-through.
Works in bull markets via buying breakouts above weekly pivot and in bear markets via selling breakdowns below weekly pivot.
Uses weekly pivot for trend direction and daily Donchian(20) for entry timing, with volume confirmation to avoid false signals.
Designed for low trade frequency (target: 50-150 trades over 4 years) to minimize fee drag.
"""

name = "6h_Donchian20_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) channels
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    high_max_20_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian(20) (20) and weekly pivot (1)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(high_max_20_aligned[i]) or 
            np.isnan(low_min_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below weekly pivot
        above_pivot = close[i] > pivot_1w_aligned[i]
        below_pivot = close[i] < pivot_1w_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-period average volume
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above Donchian high + above weekly pivot + volume
            if close[i] > high_max_20_aligned[i] and above_pivot and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + below weekly pivot + volume
            elif close[i] < low_min_20_aligned[i] and below_pivot and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or falls below weekly pivot
            if close[i] < low_min_20_aligned[i] or not above_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or rises above weekly pivot
            if close[i] > high_max_20_aligned[i] or not below_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals