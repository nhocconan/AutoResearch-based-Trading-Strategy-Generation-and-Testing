#!/usr/bin/env python3
"""
4h_CamarillaPivot_R1S1_Breakout_VolumeFilter_v1
Concept: 4h Camarilla pivot levels (R1/S1) breakout with daily volume confirmation.
- Long: Price > R1 (H4 + 1.1*(H-L)/12) AND daily volume > 1.5x 20-period average
- Short: Price < S1 (L4 - 1.1*(H-L)/12) AND daily volume > 1.5x 20-period average
- Exit: Price crosses back through daily pivot point (P = (H+L+C)/3)
- Position sizing: 0.25
- Target: 50-100 total trades over 4 years (12-25/year)
- Works in bull/bear: Volume confirms institutional interest, pivot levels provide clear support/resistance
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_CamarillaPivot_R1S1_Breakout_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily: Pivot Levels Calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    
    # Align daily pivot levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Daily: Volume Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # === 4h: Price Data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        pp = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        curr_vol = volume_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(pp) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(vol_ma) or np.isnan(curr_vol)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-period average
        vol_condition = curr_vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if close[i] > r1_val and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif close[i] < s1_val and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if close[i] < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if close[i] > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals