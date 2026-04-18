#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBreakout
6h strategy using weekly pivot points with range expansion/contraction filter.
- Long: Close breaks above weekly R1 + weekly range expansion (current week > 1.5x avg weekly range)
- Short: Close breaks below weekly S1 + weekly range expansion
- Exit: Opposite breakout or range contraction (current week < 0.5x avg weekly range)
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and range filter
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12.0
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12.0
    
    # Weekly range (high - low)
    weekly_range = high_1w - low_1w
    # Average weekly range over 4 weeks
    avg_weekly_range = pd.Series(weekly_range).rolling(window=4, min_periods=4).mean().values
    
    # Align weekly data to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    avg_range_aligned = align_htf_to_ltf(prices, df_1w, avg_weekly_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough for 4-week average
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(avg_range_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly range expansion condition
        current_week_range = high_1w[i // (7*24//6)] - low_1w[i // (7*24//6)] if i >= (7*24//6) else 0
        range_expansion = current_week_range > 1.5 * avg_range_aligned[i] if i >= (7*24//6) else False
        range_contraction = current_week_range < 0.5 * avg_range_aligned[i] if i >= (7*24//6) else False
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakdown_down = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: range expansion + breakout above weekly R1
            if range_expansion and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: range expansion + breakdown below weekly S1
            elif range_expansion and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: range contraction or breakdown below weekly S1
            if range_contraction or breakdown_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: range contraction or breakout above weekly R1
            if range_contraction or breakout_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_RangeBreakout"
timeframe = "6h"
leverage = 1.0
#%%