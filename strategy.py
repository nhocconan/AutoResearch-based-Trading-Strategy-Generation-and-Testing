#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1-day pivot levels (R1/S1, R2/S2, R3/S3) and volume confirmation.
# Fade at middle pivots (R2/S2) for mean reversion, breakout at outer pivots (R1/S1) for trend following.
# Uses daily pivot points calculated from prior day's OHLC. Works in both bull and bear markets by 
# fading extremes and catching continuations. Target: 50-150 total trades over 4 years.
name = "12h_1d_Pivot_R1S1_R2S2_Strategy"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Support and resistance levels
    s1_1d = (2 * pivot_1d) - high_1d
    r1_1d = (2 * pivot_1d) - low_1d
    s2_1d = pivot_1d - (high_1d - low_1d)
    r2_1d = pivot_1d + (high_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    
    # Align daily pivot levels to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or np.isnan(r2_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Fade at middle pivots (R2/S2) - mean reversion
            if close[i] <= s2_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] >= r2_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Breakout at outer pivots (R1/S1) - trend following
            elif close[i] > r1_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < s1_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit at R2 (take profit) or S3 (stop reversal)
            if close[i] >= r2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] <= s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit at S2 (take profit) or R3 (stop reversal)
            if close[i] <= s2_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] >= r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals