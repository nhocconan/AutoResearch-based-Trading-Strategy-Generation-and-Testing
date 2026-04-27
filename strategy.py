#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: Donchian(20) breakouts on 6h timeframe, filtered by weekly pivot point direction (bullish/bearish) and volume confirmation, capture high-probability trend continuation moves. Weekly pivot direction is derived from prior week's close vs pivot point. Volume filter requires current volume > 1.5x 20-period average. Designed to work in both bull and bear markets by using directional pivot filter. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get weekly data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot point and direction from prior week
    # Pivot = (High + Low + Close) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Direction: 1 if close > pivot (bullish), -1 if close < pivot (bearish)
    weekly_direction = np.where(close_1w > pivot_1w, 1, -1)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Align weekly direction and volume confirmation to 6h timeframe
    weekly_direction_aligned = align_htf_to_ltf(prices, df_1w, weekly_direction)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm)  # volume is 6h, but align using weekly index for signal stability
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20), weekly data (1)
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_direction_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        direction = weekly_direction_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with bullish weekly direction and volume
            if close_val > upper and direction == 1 and vol_conf:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short: price breaks below Donchian low with bearish weekly direction and volume
            elif close_val < lower and direction == -1 and vol_conf:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: price retrace to Donchian low or opposite breakout
            if close_val < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price retrace to Donchian high or opposite breakout
            if close_val > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0