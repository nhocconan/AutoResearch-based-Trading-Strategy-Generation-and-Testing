#!/usr/bin/env python3
"""
6h_1d_Pivot_Reversal_v1
Hypothesis: Fade at extreme daily pivot levels (R3/S3) with volume confirmation on 6h timeframe.
Works in bull markets by shorting overextended rallies and in bear markets by buying panic dips.
Uses 6h for execution and 1d pivots for structure, targeting low trade frequency (<30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Extreme level touches with volume confirmation
        # Long when price touches S3 with above-average volume (panic dip)
        touch_s3 = low[i] <= s3_aligned[i] * 1.001  # Allow small buffer
        vol_confirm_long = vol_ratio[i] > 1.5
        
        # Short when price touches R3 with above-average volume (overextended rally)
        touch_r3 = high[i] >= r3_aligned[i] * 0.999  # Allow small buffer
        vol_confirm_short = vol_ratio[i] > 1.5
        
        # Entry conditions
        long_entry = touch_s3 and vol_confirm_long
        short_entry = touch_r3 and vol_confirm_short
        
        # Exit conditions: price moves back toward pivot or opposite extreme touch
        long_exit = (high[i] >= pivot[i] * 0.999) or touch_r3
        short_exit = (low[i] <= pivot[i] * 1.001) or touch_s3
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals