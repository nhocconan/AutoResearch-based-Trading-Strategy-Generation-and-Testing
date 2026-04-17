#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (Standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = pivot_1d + (range_1d * 1.0)
    s1_1d = pivot_1d - (range_1d * 1.0)
    
    # Align pivot levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get weekly high/low for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    # Calculate 20-period rolling high/low on weekly data
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Volume filter: current volume > 1.3x 20-period average (less strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and above weekly 20-period high
            if close[i] > r1_1d_aligned[i] and volume_filter[i] and close[i] > high_20w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below weekly 20-period low
            elif close[i] < s1_1d_aligned[i] and volume_filter[i] and close[i] < low_20w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1
            if close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1
            if close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Weekly20_HighLow_Breakout_Volume"
timeframe = "4h"
leverage = 1.0