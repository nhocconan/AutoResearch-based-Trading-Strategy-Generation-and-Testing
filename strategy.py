#!/usr/bin/env python3
"""
4h_1d_Support_Resistance_Volume_Breakout
Hypothesis: Break above R4 or below S4 on 1-day chart with volume confirmation in 4h timeframe.
In bull markets, breaks above resistance continue upward. In bear markets, breaks below support continue downward.
Volume confirms institutional participation. Uses 1-day pivot levels for structure and 4h for entry timing.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate standard pivot points and support/resistance levels."""
    pivot = (high + low + close) / 3.0
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return pivot, r1, r2, r3, s1, s2, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    _, R1_1d, R2_1d, R3_1d, S1_1d, S2_1d, S3_1d = calculate_pivot_points(high_1d, low_1d, close_1d)
    
    # Define breakout levels: R4 and S4 (using 1.5x the pivot range)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R4_1d = pivot_1d + (range_1d * 1.5)
    S4_1d = pivot_1d - (range_1d * 1.5)
    
    # Align all data to 4h timeframe
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above R4 with volume expansion
        long_condition = (close[i] > R4_1d_aligned[i]) and volume_expansion[i]
        
        # Short breakdown: price breaks below S4 with volume expansion
        short_condition = (close[i] < S4_1d_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif long_condition and position == 1:
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        elif short_condition and position == -1:
            signals[i] = -position_size
        else:
            # Hold current position or stay flat
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Support_Resistance_Volume_Breakout"
timeframe = "4h"
leverage = 1.0