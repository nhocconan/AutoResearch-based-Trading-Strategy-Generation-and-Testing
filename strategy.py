#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_v1
Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction filter (price > weekly pivot = long bias, < weekly pivot = short bias) and volume confirmation.
Works in both bull and bear markets by using weekly pivot as regime filter (above pivot = bullish bias, below = bearish bias) and Donchian breakouts for momentum entries.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.0, ±0.25) to minimize fee drag.
"""

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
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Load weekly data for pivot points (HTF)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_vals = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above Donchian upper + price > weekly pivot (bullish bias) + volume spike
        if close[i] > highest_high[i] and close[i] > weekly_pivot_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below Donchian lower + price < weekly pivot (bearish bias) + volume spike
        elif close[i] < lowest_low[i] and close[i] < weekly_pivot_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to weekly pivot or loss of volume confirmation
        elif position == 1 and (close[i] <= weekly_pivot_aligned[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] >= weekly_pivot_aligned[i] or not volume_spike[i]):
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

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_v1"
timeframe = "6h"
leverage = 1.0