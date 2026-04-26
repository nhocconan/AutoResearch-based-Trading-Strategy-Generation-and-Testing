#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter (from 1w high/low) and volume confirmation.
Weekly pivot direction avoids counter-trend trades: long only when price > weekly midpoint, short only when price < weekly midpoint.
Volume spike confirms institutional interest. Designed for 50-150 total trades over 4 years (12-37/year) with discrete position sizing (0.0, ±0.25).
Works in both bull and bear markets by aligning with higher timeframe weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot: midpoint of weekly range
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_midpoint = (weekly_high + weekly_low) / 2.0
    
    # Align weekly midpoint to 6h timeframe
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1w, weekly_midpoint)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 20-period EMA volume from 1d (more stable)
    vol_ema_20_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period EMA volume from 1d
    volume_spike = volume > (1.5 * vol_ema_20_1d_aligned)
    
    # Calculate Donchian(20) from 6h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = max(lookback - 1, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_midpoint_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Close breaks above Donchian upper + price > weekly midpoint (uptrend bias) + volume spike
        if close[i] > highest_high[i] and close[i] > weekly_midpoint_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: Close breaks below Donchian lower + price < weekly midpoint (downtrend bias) + volume spike
        elif close[i] < lowest_low[i] and close[i] < weekly_midpoint_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: price crosses weekly midpoint in opposite direction
        elif position == 1 and close[i] < weekly_midpoint_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > weekly_midpoint_aligned[i]:
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

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0