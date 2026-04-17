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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Align weekly pivots to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Volume filter: current volume > 1.5 * 30-period average
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need weekly pivots and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or 
            np.isnan(r1_6h[i]) or 
            np.isnan(s1_6h[i]) or 
            np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or 
            np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or 
            np.isnan(volume_ma30[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma30[i])
        
        if position == 0:
            # Long: Price breaks above R1 with volume, target R2
            if (close[i] > r1_6h[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume, target S2
            elif (close[i] < s1_6h[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reaches R2 or falls back below R1
            if (close[i] >= r2_6h[i]) or (close[i] < r1_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches S2 or rises back above S1
            if (close[i] <= s2_6h[i]) or (close[i] > s1_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_R1S1_Volume"
timeframe = "6h"
leverage = 1.0