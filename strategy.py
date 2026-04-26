#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Trend_Volume
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction (from 1w) and volume confirmation.
Enters long when price breaks above Donchian(20) high with bullish weekly pivot (close > weekly pivot) and volume spike.
Enters short when price breaks below Donchian(20) low with bearish weekly pivot (close < weekly pivot) and volume spike.
Uses 1w timeframe for pivot calculation to avoid look-ahead and ensure completed bar.
Discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
Works in both bull and bear markets by requiring alignment with weekly pivot direction.
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
    
    # Calculate Donchian(20) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot: (prior week high + low + close) / 3
    prior_week_high = np.roll(df_1w['high'].values, 1)
    prior_week_low = np.roll(df_1w['low'].values, 1)
    prior_week_close = np.roll(df_1w['close'].values, 1)
    prior_week_high[0] = np.nan
    prior_week_low[0] = np.nan
    prior_week_close[0] = np.nan
    
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period Donchian)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high + bullish weekly pivot + volume spike
        if close[i] > highest_20[i] and close[i] > weekly_pivot_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below Donchian low + bearish weekly pivot + volume spike
        elif close[i] < lowest_20[i] and close[i] < weekly_pivot_aligned[i] and volume_spike[i]:
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

name = "6h_Donchian20_Breakout_WeeklyPivot_Trend_Volume"
timeframe = "6h"
leverage = 1.0