#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_Volume
# Hypothesis: Trade momentum breakouts from weekly R1/S1 levels on daily timeframe with volume confirmation.
# Weekly R1/S1 levels provide strong institutional support/resistance that hold across market cycles.
# Uses volume spike confirmation to ensure institutional participation. Works in both bull and bear markets.
# Targets 15-25 trades per year by requiring strong weekly level breaks with volume confirmation.

name = "1d_WeeklyPivot_Breakout_Volume"
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
    
    # Calculate weekly R1 and S1 levels using previous week's data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly R1 and S1 levels (key institutional levels)
    s1_1w = pivot_1w - range_1w * 1.0
    r1_1w = pivot_1w + range_1w * 1.0
    
    # Align weekly levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly R1 with volume spike
            if (close[i] > r1_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly S1 with volume spike
            elif (close[i] < s1_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below weekly S1 or momentum loss
            if close[i] < s1_aligned[i] * 0.997:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above weekly R1 or momentum loss
            if close[i] > r1_aligned[i] * 1.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals