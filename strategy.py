#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation.
# Fade at R3/S3 levels (mean reversion) and breakout continuation at R4/S4 levels.
# Uses 1d Camarilla calculation: pivot = (H+L+C)/3, range = H-L
# R4 = pivot + 1.5*range, R3 = pivot + 1.1*range, S3 = pivot - 1.1*range, S4 = pivot - 1.5*range
# Enter long when price crosses above S3 with volume > 1.5x average (mean reversion bounce)
# Enter short when price crosses below R3 with volume > 1.5x average (mean reversion fade)
# Exit on opposite Camarilla level touch or when price reaches R4/S4 (breakout continuation)
# Target: 75-200 total trades over 4 years (19-50/year) with controlled risk.

name = "6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + 1.1 * range_1d
    r4_1d = pivot_1d + 1.5 * range_1d
    s3_1d = pivot_1d - 1.1 * range_1d
    s4_1d = pivot_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price touches S4 (stop loss) or reaches R4 (take profit)
            if low[i] <= s4_1d_aligned[i] or high[i] >= r4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches R4 (stop loss) or reaches S4 (take profit)
            if high[i] >= r4_1d_aligned[i] or low[i] <= s4_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at S3/R3 with volume confirmation
            if volume[i] > volume_threshold[i]:
                # Long when price crosses above S3 (bounce from support)
                if close[i] > s3_1d_aligned[i] and close[i] < r3_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short when price crosses below R3 (fade from resistance)
                elif close[i] < r3_1d_aligned[i] and close[i] > s3_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals