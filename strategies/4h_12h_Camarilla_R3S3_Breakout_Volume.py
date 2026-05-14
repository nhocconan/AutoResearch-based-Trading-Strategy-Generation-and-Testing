# 4h_12h_Camarilla_R3S3_Breakout_Volume
# Hypothesis: Trade breakouts from 12h Camarilla R3/S3 levels on 4h timeframe with volume confirmation.
# R3/S3 levels represent stronger weekly support/resistance than R1/S1, reducing false breakouts.
# Uses volume surge (>2x 20-period average) to confirm institutional participation.
# Works in bull markets (breakouts continue) and bear markets (mean reversion from extreme levels).
# Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing to minimize fee drag.

name = "4h_12h_Camarilla_R3S3_Breakout_Volume"
timeframe = "4h"
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
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h pivot and Camarilla R3/S3 levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot point
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla R3 and S3 levels (stronger weekly support/resistance)
    s3_12h = close_12h - (range_12h * 1.1 / 4)
    r3_12h = close_12h + (range_12h * 1.1 / 4)
    
    # Align 12h levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    
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
            if (close[i] > r3_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S3 with volume surge
            elif (close[i] < s3_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S3
            if close[i] < s3_aligned[i] * 0.997:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R3
            if close[i] > r3_aligned[i] * 1.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals