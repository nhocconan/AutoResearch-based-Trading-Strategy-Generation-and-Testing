#!/usr/bin/env python3
"""
6h_1d_AngleOfAttack_Breakout
Hypothesis: Use 1-day price action to define a dynamic breakout zone (Angle of Attack - AoA).
Long when price breaks above AoA resistance with volume confirmation; short when breaks below AoA support.
AoA adapts to volatility and trend, providing robust support/resistance that works in bull/bear markets.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

name = "6h_1d_AngleOfAttack_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for AoA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1-day range and midpoint for Angle of Attack
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Angle of Attack levels: based on 1-day range
    range_1d = high_1d - low_1d
    # Avoid division by zero
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    # Calculate AoA resistance (upper) and support (lower)
    # Based on close position within the day's range
    position_in_range = (close_1d - low_1d) / range_1d  # 0 to 1
    # When price closes in upper half, resistance is higher; lower half, support is lower
    aoa_resistance = high_1d + (range_1d * 0.15 * position_in_range)
    aoa_support = low_1d - (range_1d * 0.15 * (1 - position_in_range))
    
    # Align to 6h timeframe
    aoa_resistance_aligned = align_htf_to_ltf(prices, df_1d, aoa_resistance)
    aoa_support_aligned = align_htf_to_ltf(prices, df_1d, aoa_support)
    
    # Calculate 6-day volume average for spike detection (on 6h data)
    vol_6h = prices['volume'].values
    vol_avg_6h = np.full(len(vol_6h), np.nan)
    for i in range(len(vol_6h)):
        if i >= 5:  # 6-period average
            vol_avg_6h[i] = np.mean(vol_6h[i-5:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 6  # Need enough data for 6-period volume average
    
    for i in range(start_idx, n):
        if np.isnan(aoa_resistance_aligned[i]) or np.isnan(aoa_support_aligned[i]):
            continue
            
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        vol_avg = vol_avg_6h[i]
        
        # Volume spike: current volume > 1.3x 6-period average
        vol_spike = (not np.isnan(vol_avg) and current_volume > 1.3 * vol_avg)
        
        if position == 0:
            # Long: price breaks above AoA resistance with volume spike
            if current_close > aoa_resistance_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below AoA support with volume spike
            elif current_close < aoa_support_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below AoA support or volume dries up
            if current_close < aoa_support_aligned[i] or (not np.isnan(vol_avg) and current_volume < 0.6 * vol_avg):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above AoA resistance or volume dries up
            if current_close > aoa_resistance_aligned[i] or (not np.isnan(vol_avg) and current_volume < 0.6 * vol_avg):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals