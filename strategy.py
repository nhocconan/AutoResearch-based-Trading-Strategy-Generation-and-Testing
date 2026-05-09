#!/usr/bin/env python3
name = "6H_Daily_Pivot_Range_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot and range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot point and range (high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    
    # Define breakout levels: pivot ± 0.5 * daily range
    upper_breakout = pivot_1d + 0.5 * daily_range
    lower_breakout = pivot_1d - 0.5 * daily_range
    
    # Align daily levels to 6h timeframe
    upper_breakout_aligned = align_htf_to_ltf(prices, df_1d, upper_breakout)
    lower_breakout_aligned = align_htf_to_ltf(prices, df_1d, lower_breakout)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(upper_breakout_aligned[i]) or np.isnan(lower_breakout_aligned[i]) or np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper breakout level + volume confirmation
            if close[i] > upper_breakout_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower breakout level + volume confirmation
            elif close[i] < lower_breakout_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above pivot (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals