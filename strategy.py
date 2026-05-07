#!/usr/bin/env python3
# 6H_Donchian_Breakout_WeeklyPivot_Direction_Volume
# Hypothesis: Uses 6h Donchian(20) breakouts filtered by weekly pivot direction (from 1w data) and volume confirmation.
# In uptrend (price > weekly pivot), take long breakouts above Donchian high; in downtrend (price < weekly pivot), take short breakouts below Donchian low.
# Weekly pivot provides structural bias to avoid counter-trend trades, reducing whipsaw in sideways markets.
# Target: 15-35 trades per year with size 0.25 to minimize fee drag while capturing strong directional moves.

name = "6H_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivot points
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Standard pivot point: P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channels (20-period) on 6h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + price > weekly pivot (uptrend bias) + volume spike
            if (close[i] > high_roll[i] and 
                close[i] > weekly_pivot_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + price < weekly pivot (downtrend bias) + volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < weekly_pivot_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price falls back below Donchian high or trend bias flips
            if close[i] < high_roll[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Donchian low or trend bias flips
            if close[i] > low_roll[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals