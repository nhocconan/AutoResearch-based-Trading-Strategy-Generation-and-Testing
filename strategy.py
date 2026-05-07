#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Trend
# Hypothesis: 6-hour Donchian(20) breakout with weekly pivot directional filter and volume confirmation
# Uses weekly pivot levels from Monday open to determine bias: long above weekly pivot, short below
# Combines with Donchian breakouts for momentum entries and volume filter to reduce false signals
# Designed to work in both bull and bear markets by aligning with weekly structure
# Target: 15-35 trades per year (~60-140 over 4 years) with position size 0.25

name = "6h_WeeklyPivot_DonchianBreakout_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Load weekly data ONCE for pivot calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - H
    # Resistance 1 = (2 * Pivot) - L
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot for each week using prior week's data
    weekly_pivot = np.full_like(weekly_high, np.nan)
    weekly_r1 = np.full_like(weekly_high, np.nan)
    weekly_s1 = np.full_like(weekly_high, np.nan)
    
    for i in range(1, len(weekly_high)):
        weekly_pivot[i] = (weekly_high[i-1] + weekly_low[i-1] + weekly_close[i-1]) / 3.0
        weekly_r1[i] = (2 * weekly_pivot[i]) - weekly_high[i-1]
        weekly_s1[i] = (2 * weekly_pivot[i]) - weekly_low[i-1]
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    
    # Donchian channels (20-period) on 6h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_max[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_min[i-1]  # Break below previous period's low
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # Weekly pivot bias
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + above weekly pivot
            if breakout_up and volume_confirm and above_pivot:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + below weekly pivot
            elif breakout_down and volume_confirm and below_pivot:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or crosses below weekly pivot
            if close[i] < low_min[i-1] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or crosses above weekly pivot
            if close[i] > high_max[i-1] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals