#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection
Hypothesis: On 6h timeframe, price breaking 20-period Donchian channels with weekly pivot
direction as trend filter and volume confirmation provides robust breakout signals that work
in both bull and bear markets. Weekly pivot direction (price above/below weekly pivot point)
captures the longer-term trend while avoiding whipsaws. Volume confirmation (1.5x average)
ensures breakouts have conviction. Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets
75-150 trades over 4 years (19-38/year) to stay within optimal trade frequency for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot direction (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for additional regime filter (optional)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian channels (20-period) on 6h
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume ratio (current / 20-period average) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian(20), volume MA(20)
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or
            np.isnan(high_ma[i]) or
            np.isnan(low_ma[i]) or
            np.isnan(vol_ratio[i])):
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
        vol_confirmed = vol_ratio[i] > 1.5  # volume at least 1.5x average
        
        # Donchian breakout conditions
        breakout_up = high_val > high_ma[i]   # price breaks above upper Donchian
        breakout_down = low_val < low_ma[i]   # price breaks below lower Donchian
        
        # Weekly pivot direction filter
        above_pivot = close_val > pivot_1w_aligned[i]
        below_pivot = close_val < pivot_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian AND above weekly pivot AND volume confirmation
            long_signal = breakout_up and above_pivot and vol_confirmed
            
            # Short: price breaks below lower Donchian AND below weekly pivot AND volume confirmation
            short_signal = breakout_down and below_pivot and vol_confirmed
            
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
            # Exit: price breaks below lower Donchian OR weekly pivot flips bearish
            if (low_val < low_ma[i]) or (not above_pivot):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR weekly pivot flips bullish
            if (high_val > high_ma[i]) or (not below_pivot):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection"
timeframe = "6h"
leverage = 1.0