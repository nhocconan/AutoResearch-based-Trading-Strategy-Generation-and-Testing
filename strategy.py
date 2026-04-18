#!/usr/bin/env python3
"""
6h_1d_Camarilla_Pivot_R1S1_R2S2_Breakout_Volume
Hypothesis: Uses 1d R1/S1/R2/S2 levels as a price channel. Trades breakouts of R1/S1 in the 
direction of the 1d trend (above/below daily pivot) with volume confirmation. Designed for 
both bull and bear markets by filtering with daily trend. Target: 12-37 trades/year on 6h.
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
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    r2_1d = close_1d + rng_1d * 1.1 / 6
    s2_1d = close_1d - rng_1d * 1.1 / 6
    
    # Calculate 1d pivot for trend bias
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    
    # Align all levels to 6h timeframe (wait for bar close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation: current volume > 2.0 x 24-period average (more selective)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above both 1d R1 and R2, above daily pivot, with volume
            if (close[i] > r1_1d_aligned[i] and close[i] > r2_1d_aligned[i] and 
                close[i] > pivot_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below both 1d S1 and S2, below daily pivot, with volume
            elif (close[i] < s1_1d_aligned[i] and close[i] < s2_1d_aligned[i] and 
                  close[i] < pivot_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 1d pivot or breaks below 1d S1
            if (not np.isnan(pivot_1d_aligned[i]) and close[i] < pivot_1d_aligned[i]) or \
               (not np.isnan(s1_1d_aligned[i]) and close[i] < s1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 1d pivot or breaks above 1d R1
            if (not np.isnan(pivot_1d_aligned[i]) and close[i] > pivot_1d_aligned[i]) or \
               (not np.isnan(r1_1d_aligned[i]) and close[i] > r1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_Pivot_R1S1_R2S2_Breakout_Volume"
timeframe = "6h"
leverage = 1.0