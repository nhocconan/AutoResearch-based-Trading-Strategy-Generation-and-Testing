#!/usr/bin/env python3
"""
6h_PivotRange_Reversion_WeeklyTrend_Filter
Hypothesis: Mean reversion at daily pivot levels with weekly trend filter.
In ranging markets (frequent in 2025-2026), price reverts to daily pivot (mean).
In trending markets, only take reversions aligned with weekly trend.
Designed for ~20-30 trades/year on 6h to minimize fee drag while capturing reversals.
"""

name = "6h_PivotRange_Reversion_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (using prior day)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Standard pivot: P = (H + L + C) / 3
    pivot = (high_prev + low_prev + close_prev) / 3
    # Support 1: S1 = 2*P - H
    s1 = 2 * pivot - high_prev
    # Resistance 1: R1 = 2*P - L
    r1 = 2 * pivot - low_prev
    
    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Weekly trend filter: EMA50 slope
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_slope = np.diff(ema_50, prepend=ema_50[0])
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_slope)
    
    # 6h price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily pivot (2 days), weekly EMA50 (50 bars)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(ema_50_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near S1 AND weekly uptrend (EMA50 rising)
            if low[i] <= s1_aligned[i] * 1.002 and ema_50_slope_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price near R1 AND weekly downtrend (EMA50 falling)
            elif high[i] >= r1_aligned[i] * 0.998 and ema_50_slope_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot or weekly trend turns down
            if high[i] >= pivot_aligned[i] * 0.998 or ema_50_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot or weekly trend turns up
            if low[i] <= pivot_aligned[i] * 1.002 or ema_50_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals