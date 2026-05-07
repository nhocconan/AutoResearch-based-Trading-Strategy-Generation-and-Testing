#!/usr/bin/env python3
"""
6h_Donchian_WeeklyPivot_VolumeFilter_v1
Hypothesis: Combines 6-hour Donchian(20) breakouts with weekly pivot point direction
and volume confirmation. Weekly pivots provide institutional-level support/resistance
that work in both bull and bear markets. Volume filters reduce false breakouts.
Target: 15-30 trades/year to minimize fee drag while maintaining edge.
"""

name = "6h_Donchian_WeeklyPivot_VolumeFilter_v1"
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
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly resistance 1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - df_1w['low']
    # Weekly support 1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - df_1w['high']
    
    pivot = weekly_pivot.values
    r1 = weekly_r1.values
    s1 = weekly_s1.values
    
    # Align weekly data to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND above weekly pivot with volume
            if (close[i] > high_20[i] and close[i] > pivot_aligned[i] and 
                volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND below weekly pivot with volume
            elif (close[i] < low_20[i] and close[i] < pivot_aligned[i] and 
                  volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Donchian low OR below weekly S1
            if close[i] < low_20[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Donchian high OR above weekly R1
            if close[i] > high_20[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals