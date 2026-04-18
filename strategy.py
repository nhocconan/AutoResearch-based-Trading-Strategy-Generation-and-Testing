#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_Pivot_R1S1_Breakout
Hypothesis: Uses 4-hour and daily Camarilla pivot levels (R1/S1) for directional bias and 1-hour for precise entry timing.
Trades breakouts of key support/resistance with volume confirmation to reduce false signals.
Designed to work in both bull and bear markets by capturing directional moves after consolidation.
Target: 15-35 trades/year to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and daily data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Camarilla levels (for trend bias)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    rng_4h = high_4h - low_4h
    r1_4h = close_4h + rng_4h * 1.1 / 12
    s1_4h = close_4h - rng_4h * 1.1 / 12
    
    # Calculate daily Camarilla levels (for stronger support/resistance)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Align all levels to 1h timeframe (wait for bar close)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above both 4h R1 and daily R1 with volume
            if (close[i] > r1_4h_aligned[i] and close[i] > r1_1d_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below both 4h S1 and daily S1 with volume
            elif (close[i] < s1_4h_aligned[i] and close[i] < s1_1d_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 4h pivot or breaks below 4h S1
            pivot_4h = (high_4h + low_4h + close_4h) / 3
            pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
            if (not np.isnan(pivot_4h_aligned[i]) and close[i] < pivot_4h_aligned[i]) or \
               (not np.isnan(s1_4h_aligned[i]) and close[i] < s1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price returns to 4h pivot or breaks above 4h R1
            pivot_4h = (high_4h + low_4h + close_4h) / 3
            pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
            if (not np.isnan(pivot_4h_aligned[i]) and close[i] > pivot_4h_aligned[i]) or \
               (not np.isnan(r1_4h_aligned[i]) and close[i] > r1_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h1d_Camarilla_Pivot_R1S1_Breakout"
timeframe = "1h"
leverage = 1.0