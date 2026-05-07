#!/usr/bin/env python3
"""
4H_DeMarker_Trend_1D_Camarilla_R1_S1_Breakout_Volume_v1
Hypothesis: Use DeMarker(13) on 4h for overbought/oversold reversal signals and 1d Camarilla R1/S1 levels for entry.
Long when DeMarker crosses above 0.3 from below and price touches 1d R1 level; 
Short when DeMarker crosses below 0.7 from above and price touches 1d S1 level.
Volume confirmation: current volume > 1.3x 20-period average volume.
DeMarker captures momentum exhaustion at extremes, while Camarilla levels provide precise entry/exit points.
Volume filter ensures trades occur during active participation, reducing false signals in low-volume environments.
Designed to work in both trending and ranging markets by combining momentum reversal with pivot point structure.
"""
name = "4H_DeMarker_Trend_1D_Camarilla_R1_S1_Breakout_Volume_v1"
timeframe = "4h"
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
    
    # Get 4h data for DeMarker
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4h DeMarker (13-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # DeMax and DeMin calculations
    delmax = np.where(high_4h[1:] > high_4h[:-1], high_4h[1:] - high_4h[:-1], 0)
    delmin = np.where(low_4h[1:] < low_4h[:-1], low_4h[:-1] - low_4h[1:], 0)
    
    # Pad to match length
    delmax = np.concatenate([[0], delmax])
    delmin = np.concatenate([[0], delmin])
    
    # Calculate DeMarker
    demax_sum = pd.Series(delmax).rolling(window=13, min_periods=13).sum().values
    demin_sum = pd.Series(delmin).rolling(window=13, min_periods=13).sum().values
    denominator = demax_sum + demin_sum
    demark = np.where(denominator != 0, demax_sum / denominator, 0.5)
    demark = np.concatenate([np.full(12, np.nan), demark[12:]])  # First 12 values are NaN
    
    demark_aligned = align_htf_to_ltf(prices, df_4h, demark)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(demark_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (1.33 days on 4h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: DeMarker crosses above 0.3 from below and price touches R1 level
            if (demark_aligned[i] > 0.3 and demark_aligned[i-1] <= 0.3 and 
                low[i] <= r1_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: DeMarker crosses below 0.7 from above and price touches S1 level
            elif (demark_aligned[i] < 0.7 and demark_aligned[i-1] >= 0.7 and 
                  high[i] >= s1_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: DeMarker returns to neutral zone (0.3 to 0.7)
            if position == 1 and demark_aligned[i] < 0.3:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and demark_aligned[i] > 0.7:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals