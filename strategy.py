#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_HTFTrend_Volume
Hypothesis: 6-hour Donchian(20) breakout with weekly pivot direction and 1d EMA50 trend filter, confirmed by volume spike.
Enters long when price breaks above 20-period high with bullish weekly pivot (price above weekly pivot) and bullish 1d trend.
Enters short when price breaks below 20-period low with bearish weekly pivot (price below weekly pivot) and bearish 1d trend.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 50-150 total trades over 4 years.
Works in both bull and bear markets by following the weekly pivot direction and 1d trend only.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 6h timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load weekly data for pivot direction
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    weekly_pivot_values = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_values)
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian + 50-period EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high + bullish weekly pivot + bullish 1d trend + volume spike
        if (close[i] > highest_20[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            close[i] > ema_50_1d_aligned[i] and 
            volume_spike[i]):
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below Donchian low + bearish weekly pivot + bearish 1d trend + volume spike
        elif (close[i] < lowest_20[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              close[i] < ema_50_1d_aligned[i] and 
              volume_spike[i]):
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price reverts to opposite Donchian level
        elif position == 1 and close[i] < lowest_20[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_20[i]:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_HTFTrend_Volume"
timeframe = "6h"
leverage = 1.0