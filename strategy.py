#!/usr/bin/env python3
name = "6h_Range_Breakout_with_Volume_Confirmation"
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
    
    # Get daily data for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily range (high-low) and its 20-period average
    daily_range = df_1d['high'].values - df_1d['low'].values
    range_ma20 = np.zeros(len(daily_range))
    for i in range(len(daily_range)):
        if i < 20:
            range_ma20[i] = np.mean(daily_range[:i+1]) if i > 0 else 0
        else:
            range_ma20[i] = np.mean(daily_range[i-19:i+1])
    
    # Range expansion signal: today's range > 1.5 * 20-day average range
    range_expansion = daily_range > (1.5 * range_ma20)
    
    # Direction from close vs open: bullish if close > open, bearish otherwise
    daily_direction = df_1d['close'].values > df_1d['open'].values
    
    # Align to 6h timeframe
    range_expansion_aligned = align_htf_to_ltf(prices, df_1d, range_expansion)
    daily_direction_aligned = align_htf_to_ltf(prices, df_1d, daily_direction)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(range_expansion_aligned[i]) or 
            np.isnan(daily_direction_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: range expansion + bullish daily direction + volume spike
            if (range_expansion_aligned[i] and 
                daily_direction_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: range expansion + bearish daily direction + volume spike
            elif (range_expansion_aligned[i] and 
                  not daily_direction_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: range contraction or volume drops below average
            if (not range_expansion_aligned[i] or 
                volume[i] < 0.8 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: range contraction or volume drops below average
            if (not range_expansion_aligned[i] or 
                volume[i] < 0.8 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals