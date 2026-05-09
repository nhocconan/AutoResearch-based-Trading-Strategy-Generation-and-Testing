#!/usr/bin/env python3
"""
6h_PivotPoint_Reversal_1dTrend_Volume
Hypothesis: Daily pivot reversal with volume confirmation on 6h timeframe.
In ranging markets, price tends to revert to daily pivot point (PP).
In trending markets, breaks above R1 or below S1 with volume confirm trend continuation.
Works in both bull and bear by adapting to daily pivot levels and volume.
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_PivotPoint_Reversal_1dTrend_Volume"
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])  # previous high
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])   # previous low
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]]) # previous close
    
    # Calculate daily pivot points
    pp = (ph + pl + pc) / 3.0           # Pivot Point
    r1 = 2 * pp - pl                    # Resistance 1
    s1 = 2 * pp - ph                    # Support 1
    r2 = pp + (ph - pl)                 # Resistance 2
    s2 = pp - (ph - pl)                 # Support 2
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20)  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above R1 with volume (trend continuation)
            # or price bounces off S1 with volume (mean reversion)
            if ((close[i] > r1_aligned[i] and volume_ratio[i] > 1.5) or
                (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i] and volume_ratio[i] > 1.5)):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below S1 with volume (trend continuation)
            # or price bounces off R1 with volume (mean reversion)
            elif ((close[i] < s1_aligned[i] and volume_ratio[i] > 1.5) or
                  (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i] and volume_ratio[i] > 1.5)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below PP or reaches R2
            if close[i] < pp_aligned[i] or close[i] > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above PP or reaches S2
            if close[i] > pp_aligned[i] or close[i] < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals