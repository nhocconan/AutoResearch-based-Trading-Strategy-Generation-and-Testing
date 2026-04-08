#!/usr/bin/env python3
# 6h_12h_camarilla_pivot_v1
# Hypothesis: 6-hour Camarilla pivot levels (from 12h) with volume confirmation. 
# Long: price breaks above R4 with volume surge. Short: price breaks below S4 with volume surge.
# Exit: price returns to R3/S3 or opposite breakout with volume.
# Uses 12h timeframe for pivot calculation to reduce noise and capture institutional levels.
# Works in both bull/bear markets by trading breakouts of key levels with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 12h bar: based on previous bar's H, L, C
    # R4 = C + (H-L) * 1.1/2, R3 = C + (H-L) * 1.1/4, etc.
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Initialize arrays for Camarilla levels (same length as 12h data)
    R4_12h = np.full(len(close_12h), np.nan)
    R3_12h = np.full(len(close_12h), np.nan)
    S3_12h = np.full(len(close_12h), np.nan)
    S4_12h = np.full(len(close_12h), np.nan)
    
    # Calculate for each 12h bar (starting from index 1 as we need previous bar)
    for i in range(1, len(close_12h)):
        H = high_12h[i-1]  # Previous 12h high
        L = low_12h[i-1]   # Previous 12h low
        C = close_12h[i-1] # Previous 12h close
        range_hl = H - L
        
        if range_hl <= 0:
            continue
            
        R4_12h[i] = C + range_hl * 1.1 / 2
        R3_12h[i] = C + range_hl * 1.1 / 4
        S3_12h[i] = C - range_hl * 1.1 / 4
        S4_12h[i] = C - range_hl * 1.1 / 2
    
    # Align 12h Camarilla levels to 6h timeframe
    R4_12h_aligned = align_htf_to_ltf(prices, df_12h, R4_12h)
    R3_12h_aligned = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_12h_aligned = align_htf_to_ltf(prices, df_12h, S3_12h)
    S4_12h_aligned = align_htf_to_ltf(prices, df_12h, S4_12h)
    
    # Calculate 20-period average volume for volume surge filter
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        r4 = R4_12h_aligned[i]
        r3 = R3_12h_aligned[i]
        s3 = S3_12h_aligned[i]
        s4 = S4_12h_aligned[i]
        
        # Skip if any required data is NaN
        if np.isnan(r4) or np.isnan(r3) or np.isnan(s3) or np.isnan(s4) or np.isnan(avg_vol):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol  # Volume surge filter
        
        if position == 1:  # Long position
            # Exit: price returns to R3 or reverse break below S4 with volume
            if price < r3 or (price < s4 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to S3 or reverse break above R4 with volume
            if price > s3 or (price > r4 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry: breakout of R4/S4 with volume surge
            if price > r4 and vol_surge:
                position = 1
                signals[i] = 0.25
            elif price < s4 and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals