#!/usr/bin/env python3
# 1d_1w_Pivot_R3S3_Breakout_Volume
# Hypothesis: Trade breakouts from weekly Camarilla R3/S3 levels on daily timeframe with volume confirmation.
# Weekly R3/S3 represent strong monthly support/resistance. Breakouts with volume indicate institutional interest.
# Works in bull markets (breakouts continue) and bear markets (mean reversion from extreme levels).
# Target: 10-25 trades per year by requiring precise level breaks with volume surge.

name = "1d_1w_Pivot_R3S3_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot and Camarilla R3/S3 levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla R3 and S3 levels (stronger monthly support/resistance)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    
    # Align weekly levels to daily timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R3 with volume surge
            if (close[i] > r3_aligned[i] * 1.002 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S3 with volume surge
            elif (close[i] < s3_aligned[i] * 0.998 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S3
            if close[i] < s3_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R3
            if close[i] > r3_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals