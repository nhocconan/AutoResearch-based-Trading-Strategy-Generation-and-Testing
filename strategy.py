#!/usr/bin/env python3
"""
6h_WeeklyPivot_Trend_With_Volume
Hypothesis: Use weekly pivot points from 1w data (not daily) as support/resistance. 
Enter long when price crosses above weekly pivot with weekly trend up (price > weekly EMA50) and volume spike.
Enter short when price crosses below weekly pivot with weekly trend down (price < weekly EMA50) and volume spike.
Weekly pivot provides stronger support/resistance than daily, reducing false breakouts.
Designed for 15-25 trades/year on 6h timeframe, works in bull/bear via trend filter.
"""

name = "6h_WeeklyPivot_Trend_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Standard pivot point formula
    pivot = (high_prev + low_prev + close_prev) / 3.0
    # We'll use pivot as primary S/R, with R1/S1 as secondary
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA (higher threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (needs 1w bar), EMA50 (50 bars), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA50 (uptrend) AND price crosses above weekly pivot with volume
            if close[i] > ema_50_aligned[i] and close[i] > pivot_aligned[i] and close[i-1] <= pivot_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA50 (downtrend) AND price crosses below weekly pivot with volume
            elif close[i] < ema_50_aligned[i] and close[i] < pivot_aligned[i] and close[i-1] >= pivot_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly pivot OR trend turns bearish
            if close[i] < pivot_aligned[i] and close[i-1] >= pivot_aligned[i-1] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly pivot OR trend turns bullish
            if close[i] > pivot_aligned[i] and close[i-1] <= pivot_aligned[i-1] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals