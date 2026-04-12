#!/usr/bin/env python3
"""
4h_1d_camarilla_volume_spike
Hypothesis: 4-hour strategy using daily Camarilla pivot levels with volume spike confirmation. 
Long when price touches S3 with volume spike, short when price touches R3 with volume spike.
Designed to work in both bull and bear markets by fading extremes at pivot extremes.
Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for volatility
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    
    # Pivot point (classic)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels
    r3 = pp + (high_1d - low_1d) * 1.1 / 2
    s3 = pp - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike detection (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2x 20-period average
        vol_spike = vol_ratio[i] > 2.0
        
        # Price near S3 (long) or R3 (short) with volume spike
        near_s3 = close[i] <= s3_aligned[i] * 1.005 and close[i] >= s3_aligned[i] * 0.995
        near_r3 = close[i] >= r3_aligned[i] * 0.995 and close[i] <= r3_aligned[i] * 1.005
        
        # Entry conditions
        if near_s3 and vol_spike and position != 1:
            position = 1
            signals[i] = 0.25
        elif near_r3 and vol_spike and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price moves back toward pivot point
        elif position == 1 and close[i] >= pp[i]:  # Price back above pivot
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= pp[i]:  # Price back below pivot
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_volume_spike"
timeframe = "4h"
leverage = 1.0