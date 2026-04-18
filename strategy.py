#!/usr/bin/env python3
"""
4h_12h1d_Camarilla_Pivot_R1S1_Breakout_Tight
Hypothesis: Uses 12-hour and daily Camarilla pivot levels (R1/S1) for directional bias and 4-hour for precise entry timing.
Trades breakouts of key support/resistance with volume confirmation to reduce false signals.
Designed to work in both bull and bear markets by capturing directional moves after consolidation.
Target: 10-25 trades/year to minimize fee drag while maintaining edge.
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
    
    # Get 12h and daily data for multi-timeframe analysis
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Camarilla levels (for trend bias)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    rng_12h = high_12h - low_12h
    r1_12h = close_12h + rng_12h * 1.1 / 12
    s1_12h = close_12h - rng_12h * 1.1 / 12
    
    # Calculate daily Camarilla levels (for stronger support/resistance)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Align all levels to 4h timeframe (wait for bar close)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above both 12h R1 and daily R1 with volume
            if (close[i] > r1_12h_aligned[i] and close[i] > r1_1d_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below both 12h S1 and daily S1 with volume
            elif (close[i] < s1_12h_aligned[i] and close[i] < s1_1d_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to 12h pivot or breaks below 12h S1
            pivot_12h = (high_12h + low_12h + close_12h) / 3
            pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
            if (not np.isnan(pivot_12h_aligned[i]) and close[i] < pivot_12h_aligned[i]) or \
               (not np.isnan(s1_12h_aligned[i]) and close[i] < s1_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price returns to 12h pivot or breaks above 12h R1
            pivot_12h = (high_12h + low_12h + close_12h) / 3
            pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
            if (not np.isnan(pivot_12h_aligned[i]) and close[i] > pivot_12h_aligned[i]) or \
               (not np.isnan(r1_12h_aligned[i]) and close[i] > r1_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_12h1d_Camarilla_Pivot_R1S1_Breakout_Tight"
timeframe = "4h"
leverage = 1.0