#!/usr/bin/env python3
"""
1d_WeeklyPivot_Bias_With_Trend
Hypothesis: Weekly pivot bias combined with daily trend filter captures institutional moves in both bull and bear markets.
Weekly pivot levels act as strong support/resistance. Long when price above weekly pivot and daily EMA50 up; short when below pivot and daily EMA50 down.
Uses weekly pivot (calculated from prior week) and daily EMA50 trend filter. Target: 15-25 trades/year per symbol.
"""

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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    # Using standard formula: P = (H + L + C) / 3
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot, support 1, resistance 1
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Get daily EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (need weekly data to propagate)
    start_idx = 50  # need 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema50[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot and above daily EMA50 (uptrend)
            if close[i] > pivot_aligned[i] and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot and below daily EMA50 (downtrend)
            elif close[i] < pivot_aligned[i] and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls below weekly pivot or below EMA50
            if close[i] < pivot_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above weekly pivot or above EMA50
            if close[i] > pivot_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Bias_With_Trend"
timeframe = "1d"
leverage = 1.0