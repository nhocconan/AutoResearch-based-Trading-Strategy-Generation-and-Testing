#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Breakout
6h strategy using Donchian(20) breakout with weekly pivot direction and volume confirmation.
- Long: Close breaks above 20-period high + price above weekly pivot + volume > 1.3x avg
- Short: Close breaks below 20-period low + price below weekly pivot + volume > 1.3x avg
- Exit: Opposite breakout or price crosses back below/above weekly pivot
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_max_20[i]
        breakdown_down = close[i] < low_min_20[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        # Pivot filter
        above_pivot = close[i] > weekly_pivot_aligned[i]
        below_pivot = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: breakout up + volume + above weekly pivot
            if breakout_up and vol_confirm and above_pivot:
                signals[i] = 0.25
                position = 1
            # Short: breakdown down + volume + below weekly pivot
            elif breakdown_down and vol_confirm and below_pivot:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown or price crosses below weekly pivot
            if breakdown_down or below_pivot:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout up or price crosses above weekly pivot
            if breakout_up or above_pivot:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Breakout"
timeframe = "6h"
leverage = 1.0