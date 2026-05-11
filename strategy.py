#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Trend_Follow_v1
Hypothesis: Combines weekly pivot levels (from 1w data) with 12h EMA trend filter to capture
continuation moves in both bull and bear markets. Price breaking above weekly R1 with
bullish 12h trend triggers long; breaking below weekly S1 with bearish 12h trend triggers short.
Uses weekly pivot for structure and 12h EMA for trend alignment, reducing whipsaws.
Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
"""

name = "6h_Weekly_Pivot_Trend_Follow_v1"
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
    
    # === WEEKLY DATA FOR PIVOT LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivot levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === 12H DATA FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA21 for trend
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema21_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with bullish 12h trend (price > EMA21)
            if close[i] > r1_1w_aligned[i] and ema21_12h_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with bearish 12h trend (price < EMA21)
            elif close[i] < s1_1w_aligned[i] and ema21_12h_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls back below weekly pivot OR trend turns bearish
            if close[i] < pivot_1w_aligned[i] or ema21_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises back above weekly pivot OR trend turns bullish
            if close[i] > pivot_1w_aligned[i] or ema21_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals