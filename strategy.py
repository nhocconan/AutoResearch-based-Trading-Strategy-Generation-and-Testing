#!/usr/bin/env python3
# 6h_1d_Pivot_R3S3_Reversion_with_Volume
# Hypothesis: In 6h timeframe, price tends to revert to the mean after reaching extreme 1d Camarilla levels (R3/S3).
# We fade moves beyond R3/S3 with volume confirmation, expecting mean reversion to the daily pivot.
# Works in both bull and bear markets as it captures overextended moves regardless of direction.
# Target: 15-35 trades/year by requiring multiple confirmations (extreme level + volume spike).

name = "6h_1d_Pivot_R3S3_Reversion_with_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d R3 and S3 levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R3 and S3 (extreme reversal levels)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    
    # Align 1d levels to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below S3 with volume surge (expect reversion to pivot)
            if (close[i] < s3_aligned[i] * 0.997 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above R3 with volume surge (expect reversion to pivot)
            elif (close[i] > r3_aligned[i] * 1.003 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches or crosses pivot (mean reversion complete)
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches or crosses pivot (mean reversion complete)
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals