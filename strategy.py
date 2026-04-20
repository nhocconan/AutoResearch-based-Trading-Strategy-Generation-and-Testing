#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_Volume
# Hypothesis: On 4h timeframe, trade breakouts from 12h Camarilla R1/S1 levels with volume confirmation.
# Uses 12h volume moving average to confirm breakout strength. Avoids false breakouts in low-volume environments.
# Targets 20-50 trades per year by requiring confluence of level break and volume surge.
# Works in both bull and bear markets due to price action-based entries (no directional bias).

name = "4h_Camarilla_R1S1_Breakout_Volume"
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
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    
    # Pivot point and ranges
    pivot_12h = typical_price_12h
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R1, S1
    s1_12h = close_12h - (range_12h * 1.1 / 12)
    r1_12h = close_12h + (range_12h * 1.1 / 12)
    
    # Align 12h levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long breakout above R1 with volume confirmation
            if (close[i] > r1_aligned[i] * 1.005 and 
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown below S1 with volume
            elif (close[i] < s1_aligned[i] * 0.995 and 
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below S1
            if close[i] < s1_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above R1
            if close[i] > r1_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals