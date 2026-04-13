#!/usr/bin/env python3
"""
6h_1D_1W_Pivot_Squeeze_Breakout
Hypothesis: Buy when price breaks above weekly R3 level with daily pivot support and volume expansion, sell when breaks below weekly S3 with daily resistance and volume expansion. Uses 6h timeframe with 1d/1w pivot confluence to capture institutional breakouts. Works in bull/bear markets by filtering false breakouts with multi-timeframe pivot structure and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume expansion: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    volume_expansion = volume > (vol_ma_30 * 1.8)
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high_1d = df_1d['high'].values
    prev_low_1d = df_1d['low'].values
    prev_close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pivot_1d - prev_low_1d
    s1_1d = 2 * pivot_1d - prev_high_1d
    r2_1d = pivot_1d + (prev_high_1d - prev_low_1d)
    s2_1d = pivot_1d - (prev_high_1d - prev_low_1d)
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    prev_high_1w = df_1w['high'].values
    prev_low_1w = df_1w['low'].values
    prev_close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r2_1w = pivot_1w + (prev_high_1w - prev_low_1w)
    s2_1w = pivot_1w - (prev_high_1w - prev_low_1w)
    r3_1w = prev_high_1w + 2 * (pivot_1w - prev_low_1w)
    s3_1w = prev_low_1w - 2 * (prev_high_1w - pivot_1w)
    
    # Align weekly levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Align daily levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    bars_since_entry = 0  # Track holding period
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(r2_1w_aligned[i]) or np.isnan(s2_1w_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Long setup: price above weekly R3, above daily R1, with volume expansion
        long_setup = (close[i] > r3_1w_aligned[i] and 
                     close[i] > r1_1d_aligned[i] and
                     volume_expansion[i])
        
        # Short setup: price below weekly S3, below daily S1, with volume expansion
        short_setup = (close[i] < s3_1w_aligned[i] and 
                      close[i] < s1_1d_aligned[i] and
                      volume_expansion[i])
        
        # Exit conditions: minimum holding period reached or opposite setup
        if position == 1 and bars_since_entry >= 8 and short_setup:
            position = -1
            signals[i] = -position_size
            bars_since_entry = 0
        elif position == -1 and bars_since_entry >= 8 and long_setup:
            position = 1
            signals[i] = position_size
            bars_since_entry = 0
        elif position == 0:
            if long_setup:
                position = 1
                signals[i] = position_size
                bars_since_entry = 0
            elif short_setup:
                position = -1
                signals[i] = -position_size
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1D_1W_Pivot_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0