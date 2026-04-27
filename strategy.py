#!/usr/bin/env python3
"""
#100889 - 4h_Pivot_Reversion_Volume
Hypothesis: Mean reversion to daily pivot with volume confirmation works in both bull and bear markets. 
In bull: pullbacks to pivot during uptrend. In bear: bounces from pivot during downtrend.
Uses 4h timeframe with 1d pivot and volume filter. Targets 20-30 trades/year to minimize fee drag.
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
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot (using previous day's data to avoid look-ahead)
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    # Support and resistance levels
    daily_r1 = 2 * daily_pivot - low_1d
    daily_s1 = 2 * daily_pivot - high_1d
    
    # Align to 4h timeframe (previous day's levels)
    pivot = align_htf_to_ltf(prices, df_1d, daily_pivot)
    r1 = align_htf_to_ltf(prices, df_1d, daily_r1)
    s1 = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price touches S1 support with volume, then reverses up
        if (low[i] <= s1[i] and close[i] > s1[i] and volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price touches R1 resistance with volume, then reverses down
        elif (high[i] >= r1[i] and close[i] < r1[i] and volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price reaches pivot (mean reversion complete)
        elif position == 1 and close[i] >= pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] <= pivot[i]:
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

name = "4h_Pivot_Reversion_Volume"
timeframe = "4h"
leverage = 1.0