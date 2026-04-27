#!/usr/bin/env python3
"""
#100972 - 12h_Donchian20_WeeklyPivot_Direction_Volume
Hypothesis: Donchian(20) breakout on 12h timeframe with weekly pivot trend filter and volume confirmation. 
Uses 1d high/low for weekly pivot calculation and 1w high/low for trend direction. 
Designed to capture medium-term trends with low frequency to minimize fee drag. 
Works in bull markets (breakouts with trend) and bear markets (mean reversion after false breakouts).
Target: 15-25 trades/year to stay well under fee drag limits.
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
    
    # Get 1d data for weekly pivot calculation (using prior week's high/low)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get 1w data for trend direction (weekly high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot from prior week (to avoid look-ahead)
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Calculate Donchian channels (20-period) on 12h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align weekly pivot to 12h timeframe (prior week's pivot for current period)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian high, above weekly pivot, volume spike
        if (close[i] > high_20[i] and 
            close[i] > weekly_pivot_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, below weekly pivot, volume spike
        elif (close[i] < low_20[i] and 
              close[i] < weekly_pivot_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to weekly pivot (mean reversion)
        elif position == 1 and close[i] < weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > weekly_pivot_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "12h"
leverage = 1.0