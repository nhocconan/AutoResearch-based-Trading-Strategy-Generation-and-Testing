#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeReversion_1dTrend
Hypothesis: In range-bound markets (common in 2025 BTC/ETH), price reverts to weekly pivot points.
Buy near weekly S1/S2 in uptrend (1d EMA50), sell near weekly R1/R2 in downtrend.
Uses weekly pivot as mean reversion anchor and 1d trend to avoid counter-trend trades.
Designed for 15-25 trades/year on 6f timeframe, works in sideways markets.
"""

name = "6h_WeeklyPivot_RangeReversion_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_prev = df_weekly['high'].shift(1).values
    low_prev = df_weekly['low'].shift(1).values
    close_prev = df_weekly['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 6h price
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (needs 1 week), 1d EMA50 (50 bars)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near S1/S2 AND in uptrend (above 1d EMA50)
            if close[i] > ema_50_aligned[i] and (low[i] <= s1_aligned[i] * 1.005 or low[i] <= s2_aligned[i] * 1.005):
                signals[i] = 0.25
                position = 1
            # Short: price near R1/R2 AND in downtrend (below 1d EMA50)
            elif close[i] < ema_50_aligned[i] and (high[i] >= r1_aligned[i] * 0.995 or high[i] >= r2_aligned[i] * 0.995):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot OR trend turns bearish
            if high[i] >= pivot_aligned[i] * 0.995 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot OR trend turns bullish
            if low[i] <= pivot_aligned[i] * 1.005 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals