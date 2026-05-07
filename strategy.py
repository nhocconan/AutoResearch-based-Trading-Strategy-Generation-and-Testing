#!/usr/bin/env python3
"""
12H_Camarilla_R1_S1_Breakout_1D_Trend_Filter_v1
Hypothesis: Use 12h timeframe with 1d trend filter (EMA34) and 12h Camarilla R1/S1 levels for entry.
Long when price crosses above 12h EMA34 and touches 12h R1 level; 
Short when price crosses below 12h EMA34 and touches 12h S1 level.
Volume confirmation: current volume > 1.5x 20-period average volume.
This combines trend-following with pivot point precision to reduce false signals and work in both bull and bear markets.
Target: 50-150 trades over 4 years (12-37/year) on 12h timeframe.
"""
name = "12H_Camarilla_R1_S1_Breakout_1D_Trend_Filter_v1"
timeframe = "12h"
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
    
    # Get 12h data for EMA34 trend and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    close_12h = pd.Series(df_12h['close'])
    ema34 = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34)
    
    # Calculate 12h Camarilla levels (R1, S1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r1 = pivot + (range_12h * 1.1 / 12)
    s1 = pivot - (range_12h * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (2 days on 12h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price crosses above EMA34 and touches R1 level
            if (close[i] > ema34_aligned[i] and close[i-1] <= ema34_aligned[i-1] and 
                low[i] <= r1_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price crosses below EMA34 and touches S1 level
            elif (close[i] < ema34_aligned[i] and close[i-1] >= ema34_aligned[i-1] and 
                  high[i] >= s1_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA34 side
            if position == 1 and close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals