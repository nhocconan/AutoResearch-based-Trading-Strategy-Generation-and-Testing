#!/usr/bin/env python3
# 4h_1d_Pivot_R2S2_Breakout_Volume
# Hypothesis: Trade momentum breakouts from 1d R2/S2 levels on 4h timeframe with volume confirmation.
# Uses daily pivot levels (R2/S2) as key support/resistance zones. Requires price to close beyond these levels
# with volume > 2x 20-period average. Designed for 15-35 trades per year by requiring multiple confirmations.
# Works in both bull and bear markets by capturing breakouts from key daily levels.

name = "4h_1d_Pivot_R2S2_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d R2 and S2 levels using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and range
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R2 and S2 (momentum breakout levels)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    
    # Align 1d levels to 4h timeframe
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R2 with volume surge
            if (close[i] > r2_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S2 with volume surge
            elif (close[i] < s2_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S2
            if close[i] < s2_aligned[i] * 0.997:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R2
            if close[i] > r2_aligned[i] * 1.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals