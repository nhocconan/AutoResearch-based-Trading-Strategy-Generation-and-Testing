#!/usr/bin/env python3
"""
4h_1d_Camarilla_Reversal_with_Volume
Hypothesis: Mean reversion at daily Camarilla levels (H3/L3) with volume confirmation.
In ranging markets (common in 2025), price tends to revert from extreme daily levels.
Volume filter ensures institutional interest at these levels. Works in both bull and bear
by fading extremes rather than chasing breakouts. Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    C = close
    H = high
    L = low
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    R1_1d, R2_1d, R3_1d, R4_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align all data to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: price touches S3 with volume expansion (oversold bounce)
        long_condition = (close[i] <= S3_1d_aligned[i]) and volume_expansion[i]
        
        # Short: price touches R3 with volume expansion (overbought rejection)
        short_condition = (close[i] >= R3_1d_aligned[i]) and volume_expansion[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Camarilla_Reversal_with_Volume"
timeframe = "4h"
leverage = 1.0