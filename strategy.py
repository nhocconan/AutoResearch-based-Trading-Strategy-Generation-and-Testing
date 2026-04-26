#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts in direction of weekly pivot trend (price above/below weekly pivot point). Weekly pivot acts as institutional trend filter. Donchian breakouts capture momentum. Works in bull (long when price > weekly pivot + breakout above Donchian high) and bear (short when price < weekly pivot + breakout below Donchian low). Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot point (P) = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe (extra delay not needed as based on completed weekly bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_1w)
    
    # Donchian channels (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of weekly pivot (needs 1), Donchian (20)
    start_idx = max(1, lookback) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        pivot_val = weekly_pivot_aligned[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        
        if position == 0:
            # Long: Price above weekly pivot AND break above Donchian high
            long_signal = (close_val > pivot_val) and (high_val > donchian_high)
            
            # Short: Price below weekly pivot AND break below Donchian low
            short_signal = (close_val < pivot_val) and (low_val < donchian_low)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Close below weekly pivot (trend change) OR price re-enters Donchian channel (breakout failed)
            if (close_val < pivot_val) or (close_val < donchian_high):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Close above weekly pivot (trend change) OR price re-enters Donchian channel (breakdown failed)
            if (close_val > pivot_val) or (close_val > donchian_low):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian_Breakout_v1"
timeframe = "6h"
leverage = 1.0