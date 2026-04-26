#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_v1
Hypothesis: 6h Donchian(20) breakout strategy filtered by weekly pivot direction and volume confirmation.
- Uses 6h timeframe for lower trade frequency (target: 50-150 total trades over 4 years = 12-37/year)
- Donchian breakout calculated from 6h high/low of previous 20 bars
- Weekly pivot direction (from 1w HTF) determines bias: only long above weekly pivot, only short below
- Volume spike (2x 20-period average) confirms breakout strength
- Designed for 12-37 trades/year to minimize fee drag while capturing significant moves
- Works in bull/bear markets by aligning with weekly structure and using Donchian for precise entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate Donchian channels from 6h data (20-period high/low)
    # We'll compute this directly from prices as it's our primary timeframe
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian and volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous 20-period high
        breakout_down = close[i] < low_20[i-1]  # Break below previous 20-period low
        
        # Weekly pivot filter
        above_weekly_pivot = close[i] > weekly_pivot_aligned[i]
        below_weekly_pivot = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly pivot AND volume spike
            if breakout_up and above_weekly_pivot and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below weekly pivot AND volume spike
            elif breakout_down and below_weekly_pivot and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR weekly pivot
            if close[i] < low_20[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR weekly pivot
            if close[i] > high_20[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_v1"
timeframe = "6h"
leverage = 1.0