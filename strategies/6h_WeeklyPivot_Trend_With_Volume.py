#!/usr/bin/env python3
"""
6h_WeeklyPivot_Trend_With_Volume
Hypothesis: Use weekly pivot levels (P, R1, S1) for 6h trend continuation entries, filtered by 12h EMA trend and volume confirmation. 
Weekly pivots provide strong institutional levels that work in both bull and bear markets via trend filter.
Designed for 15-25 trades/year, avoids overtrading while capturing major moves.
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
    
    # Get weekly data for pivot calculation and 12h data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_weekly) < 5 or len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_prev = df_weekly['high'].shift(1).values
    low_prev = df_weekly['low'].shift(1).values
    close_prev = df_weekly['close'].shift(1).values
    
    # Standard pivot: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_p = (high_prev + low_prev + close_prev) / 3
    pivot_r1 = 2 * pivot_p - low_prev
    pivot_s1 = 2 * pivot_p - high_prev
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pivot_p_aligned = align_htf_to_ltf(prices, df_weekly, pivot_p)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_weekly, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_weekly, pivot_s1)
    
    # 12h EMA50 for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Get 6h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (needs 1 week), EMA50 (50 bars), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(pivot_p_aligned[i]) or 
            np.isnan(pivot_r1_aligned[i]) or
            np.isnan(pivot_s1_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA50 (uptrend) AND price breaks above weekly R1 with volume
            if close[i] > ema_50_aligned[i] and high[i] > pivot_r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA50 (downtrend) AND price breaks below weekly S1 with volume
            elif close[i] < ema_50_aligned[i] and low[i] < pivot_s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly pivot P OR trend turns bearish
            if low[i] < pivot_p_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly pivot P OR trend turns bullish
            if high[i] > pivot_p_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals