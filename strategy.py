#!/usr/bin/env python3
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
    
    # Daily high/low for pivot calculation (from previous day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points and levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot_1d - low_1d
    s1 = 2 * pivot_1d - high_1d
    r2 = pivot_1d + (high_1d - low_1d)
    s2 = pivot_1d - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot_1d - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align daily pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r2 + (high_1d - low_1d))  # R4 = R2 + range
    s4_aligned = align_htf_to_ltf(prices, df_1d, s2 - (high_1d - low_1d))  # S4 = S2 - range
    
    # Volume confirmation: current > 1.5x median of last 24 bars (4 days)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=24).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long conditions: price breaks above R4 with volume OR bounces from S3 with volume
        long_breakout = (close[i] > r4_aligned[i] and volume[i] > vol_threshold[i])
        long_bounce = (close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1] and 
                      volume[i] > vol_threshold[i])
        
        # Short conditions: price breaks below S4 with volume OR rejected from R3 with volume
        short_breakdown = (close[i] < s4_aligned[i] and volume[i] > vol_threshold[i])
        short_rejection = (close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1] and 
                          volume[i] > vol_threshold[i])
        
        # Exit conditions: price returns to pivot level
        exit_long = (signals[i-1] > 0 and close[i] < pivot_1d_aligned[i])
        exit_short = (signals[i-1] < 0 and close[i] > pivot_1d_aligned[i])
        
        if long_breakout or long_bounce:
            signals[i] = 0.25
        elif short_breakdown or short_rejection:
            signals[i] = -0.25
        elif exit_long or exit_short:
            signals[i] = 0.0
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Camarilla_R3S3_R4S4_Volume"
timeframe = "6h"
leverage = 1.0