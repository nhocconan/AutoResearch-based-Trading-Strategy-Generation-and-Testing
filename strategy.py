#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_WeeklyPivotFilter_v1
Hypothesis: On 6h timeframe, trade Donchian(20) breakouts filtered by 1d EMA50 trend and weekly pivot position.
Only take longs when price > weekly pivot and price > 1d EMA50, shorts when price < weekly pivot and price < 1d EMA50.
Weekly pivot acts as regime filter (bull/bear bias), 1d EMA50 as intermediate trend, Donchian breakout for entry timing.
Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drift.
Works in bull/bear markets via weekly pivot bias and 1d trend filter.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate weekly pivot (standard: (H+L+C)/3) from prior weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian(20) channels on 6h
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), Donchian (20)
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or
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
        
        ema_50_val = ema_50_1d_aligned[i]
        pivot_val = pivot_1w_aligned[i]
        close_val = close[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, above weekly pivot, above 1d EMA50
            long_signal = (close_val > highest_high_val) and (close_val > pivot_val) and (close_val > ema_50_val)
            
            # Short: price breaks below Donchian low, below weekly pivot, below 1d EMA50
            short_signal = (close_val < lowest_low_val) and (close_val < pivot_val) and (close_val < ema_50_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR weekly pivot
            if (close_val < lowest_low_val) or (close_val < pivot_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR weekly pivot
            if (close_val > highest_high_val) or (close_val > pivot_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_WeeklyPivotFilter_v1"
timeframe = "6h"
leverage = 1.0