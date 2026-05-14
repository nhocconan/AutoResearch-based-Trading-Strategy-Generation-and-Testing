#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_R1S1_Breakout_Volume_v2
Hypothesis: Trade breakouts of 12h R1/S1 levels with volume confirmation and 12h trend bias.
Uses tighter volume filter (3x average) and stricter exit to reduce trades. 
Designed for both bull/bear markets via 12h trend filter. Target: 20-40 trades/year.
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
    
    # Get 12h data for multi-timeframe analysis
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    rng_12h = high_12h - low_12h
    r1_12h = close_12h + rng_12h * 1.1 / 12
    s1_12h = close_12h - rng_12h * 1.1 / 12
    
    # Calculate 12h pivot for trend bias
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    
    # Align all levels to 4h timeframe (wait for bar close)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # Volume confirmation: current volume > 3.0 x 30-period average (more selective)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    vol_confirm = volume > (vol_ma * 3.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(pivot_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above both 12h R1 and above 12h pivot, with volume
            if (close[i] > r1_12h_aligned[i] and 
                close[i] > pivot_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below both 12h S1 and below 12h pivot, with volume
            elif (close[i] < s1_12h_aligned[i] and 
                  close[i] < pivot_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 12h S1 (more conservative) or breaks below 12h S1
            if (not np.isnan(s1_12h_aligned[i]) and close[i] < s1_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 12h R1 (more conservative) or breaks above 12h R1
            if (not np.isnan(r1_12h_aligned[i]) and close[i] > r1_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_Pivot_R1S1_Breakout_Volume_v2"
timeframe = "4h"
leverage = 1.0