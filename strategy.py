#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly pivot: use 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Resistance 1: 2*P - L
    r1_1w = 2 * pivot_1w - low_1w
    # Support 1: 2*P - H
    s1_1w = 2 * pivot_1w - high_1w
    
    # Align weekly pivots to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # 6-period EMA on 1d for trend filter
    ema6_1d = pd.Series(close_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    ema6_1d_aligned = align_htf_to_ltf(prices, df_1d, ema6_1d)
    
    # 6h EMA13 for entry timing
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema6_1d_aligned[i]) or 
            np.isnan(ema13_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot and R1, uptrend (EMA6 rising)
            if (close[i] > pivot_1w_aligned[i] and 
                close[i] > r1_1w_aligned[i] and
                ema6_1d_aligned[i] > ema6_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot and S1, downtrend (EMA6 falling)
            elif (close[i] < pivot_1w_aligned[i] and 
                  close[i] < s1_1w_aligned[i] and
                  ema6_1d_aligned[i] < ema6_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly pivot
            if close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly pivot
            if close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals