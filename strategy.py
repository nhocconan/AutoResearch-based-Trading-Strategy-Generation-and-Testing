#!/usr/bin/env python3
"""
6h_12h_Weekly_Pivot_Donchian_Breakout
Hypothesis: Combines weekly pivot levels (direction) with 6-hour Donchian(20) breakouts and volume confirmation.
Weekly pivots provide institutional support/resistance; breakouts with volume capture momentum.
Works in bull markets (breakouts up) and bear markets (breakdowns down) by using pivot bias.
Targets 100-200 total trades over 4 years (25-50/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Weekly_Pivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H+L+C)/3
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    pivot_w = (high_w + low_w + close_w) / 3.0
    
    # Weekly bias: price above/below pivot
    bias_up = close_w > pivot_w
    bias_down = close_w < pivot_w
    
    # Align weekly bias to 6h timeframe (wait for weekly close)
    bias_up_aligned = align_htf_to_ltf(prices, df_w, bias_up.astype(float))
    bias_down_aligned = align_htf_to_ltf(prices, df_w, bias_down.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(bias_up_aligned[i]) or 
            np.isnan(bias_down_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_max[i-1]  # Break above previous high
        breakdown_down = close[i] < low_min[i-1]  # Break below previous low
        
        # Entry conditions: trade in direction of weekly bias
        long_entry = breakout_up and volume_filter and bias_up_aligned[i] > 0.5
        short_entry = breakdown_down and volume_filter and bias_down_aligned[i] > 0.5
        
        # Exit conditions: opposite Donchian break or loss of bias
        long_exit = (close[i] < low_min[i-1]) or (bias_up_aligned[i] < 0.5)
        short_exit = (close[i] > high_max[i-1]) or (bias_down_aligned[i] < 0.5)
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals