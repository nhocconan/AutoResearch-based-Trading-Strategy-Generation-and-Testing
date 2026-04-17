#!/usr/bin/env python3
"""
1h_DailyPivot_Breakout_VolumeFilter
Hypothesis: 1-hour breakouts above/below daily pivot levels (R1/S1) with volume confirmation.
In bull markets, buy breakouts above R1; in bear markets, sell breakdowns below S1.
Volume filter ensures conviction. Uses daily pivot from prior day (no look-ahead).
Trades only during active session (08-20 UTC) to avoid low-volume noise.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 1h timeframe (use previous day's levels)
    pivot_1h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need sufficient data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1h[i]) or np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume
            if close[i] > r1_1h[i] and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below S1 with volume
            elif close[i] < s1_1h[i] and volume_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls below pivot
            if close[i] < pivot_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises above pivot
            if close[i] > pivot_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DailyPivot_Breakout_VolumeFilter"
timeframe = "1h"
leverage = 1.0