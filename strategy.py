#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_VolumeConfirmation_v1
Hypothesis: Use weekly pivot levels (R1/S1) as dynamic support/resistance.
Enter long when price breaks above weekly R1 with volume confirmation (volume > 1.5x average).
Enter short when price breaks below weekly S1 with volume confirmation.
Exit when price crosses back below weekly pivot point (long) or above pivot point (short).
Use 6h timeframe for execution, weekly pivot for structure.
Target: 50-150 total trades over 4 years (12-37/year) with volume filter reducing noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points (from weekly data) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate pivot points using weekly OHLC
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe (with 1-bar delay for completed weekly bar)
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # === Volume Filter (6h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly R1 with volume confirmation
            if (close[i] > r1_w_aligned[i] and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly S1 with volume confirmation
            elif (close[i] < s1_w_aligned[i] and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses back below weekly pivot point
            if close[i] < pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back above weekly pivot point
            if close[i] > pivot_w_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0