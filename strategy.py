#!/usr/bin/env python3
"""
6h_Pivot_R1_S1_Breakout_VolumeFilter
Hypothesis: Use daily pivot points (R1/S1) as dynamic support/resistance levels. 
Breakouts above R1 with volume confirmation indicate bullish momentum; breakdowns below S1 indicate bearish momentum. 
This strategy works in both bull and bear markets because it captures momentum bursts regardless of direction, 
and volume confirmation ensures we only trade during periods of institutional participation. 
Daily pivots adapt to recent price action, making them effective across regimes. 
Targeting 15-30 trades/year to minimize fee drag.
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
    
    # === Daily Pivot Points (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Classic pivot point calculation
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2.0 * pivot_1d - low_1d
    s1_1d = 2.0 * pivot_1d - high_1d
    
    # Align daily pivot levels to 6h
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Volume Filter: 20-period average ===
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-period volume average
    warmup = 20
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above R1 with volume
            if close[i] > r1_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume
            elif close[i] < s1_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse at opposite level
        elif position == 1:
            if close[i] < s1_1d_aligned[i]:  # reverse to short if breaks S1
                signals[i] = -0.25
                position = -1
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > r1_1d_aligned[i]:  # reverse to long if breaks R1
                signals[i] = 0.25
                position = 1
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0