#!/usr/bin/env python3
"""
6h_WeeklyPivot_Directional_Bias
Hypothesis: Use weekly pivot points to establish directional bias (above/below weekly pivot), then trade breakouts of daily support/resistance levels in the direction of the weekly bias on 6h timeframe. Weekly pivot provides higher-timeframe structure that works in both bull and bear markets by defining the major trend context. Daily S1/R1 levels provide precise entry points with favorable risk-reward. Target: 20-30 trades/year.
"""

name = "6h_WeeklyPivot_Directional_Bias"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot points (directional bias)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_prev_week = df_1w['high'].shift(1).values
    low_prev_week = df_1w['low'].shift(1).values
    close_prev_week = df_1w['close'].shift(1).values
    
    # Weekly pivot point calculation
    weekly_pivot = (high_prev_week + low_prev_week + close_prev_week) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_prev_week
    weekly_s1 = 2 * weekly_pivot - high_prev_week
    weekly_r2 = weekly_pivot + (high_prev_week - low_prev_week)
    weekly_s2 = weekly_pivot - (high_prev_week - low_prev_week)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # Get daily data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    high_prev_day = df_1d['high'].shift(1).values
    low_prev_day = df_1d['low'].shift(1).values
    close_prev_day = df_1d['close'].shift(1).values
    
    # Daily pivot point calculation
    daily_pivot = (high_prev_day + low_prev_day + close_prev_day) / 3.0
    daily_r1 = 2 * daily_pivot - low_prev_day
    daily_s1 = 2 * daily_pivot - high_prev_day
    
    # Align daily levels to 6h timeframe
    daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Get 6h data for price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly data (shifted) and daily data (shifted)
    start_idx = 20  # reasonable warmup for EMA and data alignment
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or
            np.isnan(daily_r1_aligned[i]) or
            np.isnan(daily_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly bias: price vs weekly pivot
        weekly_bias_up = close[i] > weekly_pivot_aligned[i]
        weekly_bias_down = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:
            # Long: price above weekly pivot AND breaks above daily R1 with volume
            if weekly_bias_up and high[i] > daily_r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND breaks below daily S1 with volume
            elif weekly_bias_down and low[i] < daily_s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR daily S1
            if low[i] < weekly_pivot_aligned[i] or low[i] < daily_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR daily R1
            if high[i] > weekly_pivot_aligned[i] or high[i] > daily_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals