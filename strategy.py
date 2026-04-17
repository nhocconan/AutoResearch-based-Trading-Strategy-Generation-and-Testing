#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_VolumeFilter_V2
Hypothesis: Use weekly pivot levels (R1/S1) as dynamic support/resistance and breakout of 6h Donchian(20) for entry, with volume confirmation. Weekly pivot provides weekly structure, Donchian breakout captures momentum, and volume filter ensures conviction. Designed for 20-40 trades/year to work in both bull/bear regimes by aligning with higher timeframe pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for pivot points ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === 6h Donchian channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    vol_avg20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers Donchian and volume
    warmup = 20
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg20[i]) or 
            np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        vol_filter = volume[i] > 1.3 * vol_avg20[i]
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above Donchian high AND above weekly R1
            if close[i] > donchian_high[i] and close[i] > r1_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below Donchian low AND below weekly S1
            elif close[i] < donchian_low[i] and close[i] < s1_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse when price returns to weekly pivot area
        elif position == 1:
            if close[i] < pivot_1w_aligned[i]:  # exit long when price breaks below weekly pivot
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > pivot_1w_aligned[i]:  # exit short when price breaks above weekly pivot
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeFilter_V2"
timeframe = "6h"
leverage = 1.0