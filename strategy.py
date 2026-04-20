#!/usr/bin/env python3
# 1h_4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume
# Hypothesis: Trade breakouts from 4h/1d Camarilla R1/S1 levels on 1h timeframe with volume confirmation.
# Uses 4h and 1d pivot levels for institutional reference points, volume surge for confirmation.
# Designed for 15-37 trades per year by requiring precise level breaks with volume surge.
# Works in bull markets (breakouts continue) and bear markets (mean reversion from extreme levels).
# Uses session filter (08-20 UTC) to reduce noise trades.

name = "1h_4h_1d_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "1h"
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
    
    # Get 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h pivot and Camarilla R1/S1 levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    s1_4h = close_4h - (range_4h * 1.1 / 12)
    r1_4h = close_4h + (range_4h * 1.1 / 12)
    
    # Calculate 1d pivot and Camarilla R1/S1 levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    
    # Align 4h and 1d levels to 1h timeframe
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(s1_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or
            not (8 <= hours[i] <= 20)):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above both 4h R1 and 1d R1 with volume surge
            if (close[i] > r1_4h_aligned[i] * 1.002 and 
                close[i] > r1_1d_aligned[i] * 1.002 and
                volume[i] > 2.0 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below both 4h S1 and 1d S1 with volume surge
            elif (close[i] < s1_4h_aligned[i] * 0.998 and 
                  close[i] < s1_1d_aligned[i] * 0.998 and
                  volume[i] > 2.0 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price below 4h S1 or 1d S1
            if close[i] < s1_4h_aligned[i] * 0.998 or close[i] < s1_1d_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price above 4h R1 or 1d R1
            if close[i] > r1_4h_aligned[i] * 1.002 or close[i] > r1_1d_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals