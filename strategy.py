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
    
    # 12h high/low for pivot points
    daily = get_htf_data(prices, '12h')
    high_12h = daily['high'].values
    low_12h = daily['low'].values
    close_12h = daily['close'].values
    
    # Calculate pivot points for 12h
    # Pivot = (H + L + C) / 3
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    # R1 = 2*P - L
    r1_12h = 2 * pivot_12h - low_12h
    # S1 = 2*P - H
    s1_12h = 2 * pivot_12h - high_12h
    # R2 = P + (H - L)
    r2_12h = pivot_12h + (high_12h - low_12h)
    # S2 = P - (H - L)
    s2_12h = pivot_12h - (high_12h - low_12h)
    # R3 = H + 2*(P - L)
    r3_12h = high_12h + 2 * (pivot_12h - low_12h)
    # S3 = L - 2*(H - P)
    s3_12h = low_12h - 2 * (high_12h - pivot_12h)
    
    # Align pivot levels to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, daily, pivot_12h)
    r1_12h_aligned = align_htf_to_ltf(prices, daily, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, daily, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, daily, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, daily, s2_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, daily, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, daily, s3_12h)
    
    # Volume filter: 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(r2_12h_aligned[i]) or 
            np.isnan(s2_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long conditions: price above S1 with volume
        if (close[i] > s1_12h_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short conditions: price below R1 with volume
        elif (close[i] < r1_12h_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses pivot in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < pivot_12h_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > pivot_12h_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_12h_Pivot_R1S1_VolumeFilter"
timeframe = "6h"
leverage = 1.0