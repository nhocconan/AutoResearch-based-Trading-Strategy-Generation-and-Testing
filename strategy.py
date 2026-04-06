#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Filter
Hypothesis: Combines medium-term breakout with weekly structural bias to capture trends in both bull and bear markets.
Weekly pivot provides directional filter, while volume confirms institutional participation.
Designed for 6h timeframe with target of 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weeklypivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard floor trader pivots)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Align weekly pivot data to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = 20  # For Donchian channel
    
    for i in range(start, n):
        # Skip if required weekly data not available
        if np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR weekly S3 broken
            if close[i] < lowest_low or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR weekly R3 broken
            if close[i] > highest_high or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Weekly pivot bias: long above R3, short below S3
            weekly_long_bias = close[i] > r3_aligned[i]
            weekly_short_bias = close[i] < s3_aligned[i]
            
            if i >= 20 and bull_breakout and volume_filter and weekly_long_bias:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and weekly_short_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals