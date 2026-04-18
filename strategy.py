#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_S1_S3_Bounce"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    h1d = df_1d['high'].values
    l1d = df_1d['low'].values
    c1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # S1 = C - (H-L)*1.05/4
    # S3 = C - (H-L)*1.05/2
    # R1 = C + (H-L)*1.05/4
    # R3 = C + (H-L)*1.05/2
    range_1d = h1d - l1d
    s1_1d = c1d - range_1d * 1.05 / 4
    s3_1d = c1d - range_1d * 1.05 / 2
    r1_1d = c1d + range_1d * 1.05 / 4
    r3_1d = c1d + range_1d * 1.05 / 2
    
    # Align pivots to 12h (wait for daily close)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # Volume filter: current volume > 1.5 * 24-period average (2 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s1_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(r1_12h[i]) or 
            np.isnan(r3_12h[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        s1_val = s1_12h[i]
        s3_val = s3_12h[i]
        r1_val = r1_12h[i]
        r3_val = r3_12h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long conditions:
            # 1. Price near S1 support (within 0.5%)
            # 2. Price near S3 support (within 0.5%)
            # 3. Volume confirmation
            near_s1 = abs(close_val - s1_val) / s1_val < 0.005
            near_s3 = abs(close_val - s3_val) / s3_val < 0.005
            if (near_s1 or near_s3) and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Price near R1 resistance (within 0.5%)
            # 2. Price near R3 resistance (within 0.5%)
            # 3. Volume confirmation
            near_r1 = abs(close_val - r1_val) / r1_val < 0.005
            near_r3 = abs(close_val - r3_val) / r3_val < 0.005
            if (near_r1 or near_r3) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price moves to midpoint or hits resistance
            midpoint = (s1_val + r1_val) / 2
            if close_val > midpoint or close_val > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price moves to midpoint or hits support
            midpoint = (s1_val + r1_val) / 2
            if close_val < midpoint or close_val < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals