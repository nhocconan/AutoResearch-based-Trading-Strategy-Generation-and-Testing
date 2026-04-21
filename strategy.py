#!/usr/bin/env python3
"""
6h_1d_WeeklyPivot_Trend_Breakout
Hypothesis: Use 1d and 1w pivot levels as trend filters for 6h breakouts.
Long when price breaks above 1d R1 and is above 1w pivot point.
Short when price breaks below 1d S1 and is below 1w pivot point.
Exit when price crosses back through the 1d pivot point.
Weekly pivot provides stronger trend context to reduce false breakouts in both bull and bear markets.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data once for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d Pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        pp_1d[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1_1d[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
        s1_1d[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
    
    # Shift to align with current day (levels are based on previous day)
    pp_1d = np.roll(pp_1d, 1)
    r1_1d = np.roll(r1_1d, 1)
    s1_1d = np.roll(s1_1d, 1)
    pp_1d[0] = np.nan
    r1_1d[0] = np.nan
    s1_1d[0] = np.nan
    
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1w Pivot point (based on previous week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(high_1w)):
        pp_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
    
    # Shift to align with current week (levels are based on previous week)
    pp_1w = np.roll(pp_1w, 1)
    pp_1w[0] = np.nan
    
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(pp_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        if position == 0:
            # Long conditions: break above 1d R1 and price above 1w pivot
            if price > r1_1d_aligned[i] and price > pp_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below 1d S1 and price below 1w pivot
            elif price < s1_1d_aligned[i] and price < pp_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below 1d pivot point
            if price < pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above 1d pivot point
            if price > pp_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_WeeklyPivot_Trend_Breakout"
timeframe = "6h"
leverage = 1.0