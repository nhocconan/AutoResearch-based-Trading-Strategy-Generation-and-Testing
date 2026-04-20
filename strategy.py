#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_Volume
# Hypothesis: On daily timeframe, trade breakouts from weekly Camarilla R1/S1 levels with volume confirmation.
# Uses weekly volume moving average to confirm breakout strength. Targets 10-25 trades per year.
# Works in both bull and bear markets due to price action-based entries (no directional bias).

name = "1d_WeeklyPivot_Breakout_Volume"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    
    # Pivot point and ranges
    pivot_1w = typical_price_1w
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R1, S1 (inner levels for breakout)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    
    # Align weekly levels to daily timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Volume average for spike detection (10-period weekly aligned)
    volume_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above R1 with volume confirmation
            if (close[i] > r1_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below S1 with volume
            elif (close[i] < s1_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below S1
            if close[i] < s1_aligned[i] * 0.997:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above R1
            if close[i] > r1_aligned[i] * 1.003:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals