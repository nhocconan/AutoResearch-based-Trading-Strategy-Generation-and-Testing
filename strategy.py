#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_WeeklyPivot_Direction"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly pivot point calculation
    prev_weekly_high = np.roll(df_1w['high'].values, 1)
    prev_weekly_low = np.roll(df_1w['low'].values, 1)
    prev_weekly_close = np.roll(df_1w['close'].values, 1)
    prev_weekly_high[0] = df_1w['high'].values[0]
    prev_weekly_low[0] = df_1w['low'].values[0]
    prev_weekly_close[0] = df_1w['close'].values[0]
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly trend: price above/below weekly pivot
    weekly_trend = (df_1w['close'].values > weekly_pivot).astype(float)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Daily Donchian(20) for entry signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(weekly_trend_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above daily Donchian high AND weekly trend is up (price > weekly pivot)
            long_cond = (close[i] > high_20_aligned[i] and weekly_trend_aligned[i] > 0.5)
            
            # Short: price breaks below daily Donchian low AND weekly trend is down (price < weekly pivot)
            short_cond = (close[i] < low_20_aligned[i] and weekly_trend_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below daily Donchian low
            if close[i] < low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above daily Donchian high
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s strategy using daily Donchian(20) breakouts filtered by weekly pivot direction.
# Enters long when price breaks above 20-day high AND weekly trend is up (price above weekly pivot).
# Enters short when price breaks below 20-day low AND weekly trend is down (price below weekly pivot).
# Uses weekly pivot as trend filter to avoid counter-trend trades. Exit on opposite Donchian break.
# Designed for 6h timeframe with target 50-150 total trades over 4 years (12-37/year).
# Works in both bull/bear markets by aligning with weekly trend direction. Uses discrete sizing (0.25) to minimize churn.