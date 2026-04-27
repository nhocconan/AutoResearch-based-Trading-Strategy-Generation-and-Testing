#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: Trade Donchian(20) breakouts on 6h only when aligned with weekly pivot bias (price above/below weekly pivot) and volume confirmation. Weekly pivot provides institutional bias that works in both bull (breakouts with bias) and bear (fade against bias when overextended). Target 15-25 trades/year to avoid fee drag.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Pivot Point (standard calculation)
    weekly_high = df_1w['high']
    weekly_low = df_1w['low']
    weekly_close = df_1w['close']
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align weekly pivot to 6h (use previous week's pivot)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    
    # Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Donchian and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            continue
        
        pivot_level = pivot_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly pivot + volume spike
            if close[i] > upper_channel and close[i] > pivot_level and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + below weekly pivot + volume spike
            elif close[i] < lower_channel and close[i] < pivot_level and vol_spike_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or falls below weekly pivot
            if close[i] < lower_channel or close[i] < pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or rises above weekly pivot
            if close[i] > upper_channel or close[i] > pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0