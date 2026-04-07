#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_filter_v1
Hypothesis: Camarilla pivot levels from 1d timeframe with volume confirmation on 12h.
Enter long when price touches S3 level with volume > 20-period average.
Enter short when price touches R3 level with volume > 20-period average.
Exit when price moves back to P level or opposite S/R level.
Works in both bull and bear markets by capturing mean reversion at extreme levels.
Target: 15-30 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_filter_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    pivot = (high + low + close) / 3
    range_val = high - low
    s3 = close - (range_val * 1.1 / 2)
    s2 = close - (range_val * 1.1 / 4)
    s1 = close - (range_val * 1.1 / 6)
    r1 = close + (range_val * 1.1 / 6)
    r2 = close + (range_val * 1.1 / 4)
    r3 = close + (range_val * 1.1 / 2)
    return pivot, s1, s2, s3, r1, r2, r3

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar
    pivots = np.full(len(df_1d), np.nan)
    s1 = np.full(len(df_1d), np.nan)
    s2 = np.full(len(df_1d), np.nan)
    s3 = np.full(len(df_1d), np.nan)
    r1 = np.full(len(df_1d), np.nan)
    r2 = np.full(len(df_1d), np.nan)
    r3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        p, s1_, s2_, s3_, r1_, r2_, r3_ = calculate_camarilla(
            df_1d['high'].iloc[i], 
            df_1d['low'].iloc[i], 
            df_1d['close'].iloc[i]
        )
        pivots[i] = p
        s1[i] = s1_
        s2[i] = s2_
        s3[i] = s3_
        r1[i] = r1_
        r2[i] = r2_
        r3[i] = r3_
    
    # Align Camarilla levels to 12h timeframe
    pivots_aligned = align_htf_to_ltf(prices, df_1d, pivots)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(pivots_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to pivot level or reaches S1
            if close[i] >= pivots_aligned[i] or close[i] <= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot level or reaches R1
            if close[i] <= pivots_aligned[i] or close[i] >= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches S3 level with volume confirmation
            if (close[i] <= s3_aligned[i] * 1.001 and close[i] >= s3_aligned[i] * 0.999) and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price touches R3 level with volume confirmation
            elif (close[i] >= r3_aligned[i] * 0.999 and close[i] <= r3_aligned[i] * 1.001) and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals