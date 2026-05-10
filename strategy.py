#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Trend_Volume
Hypothesis: Use weekly pivot points (R2/S2) for breakout entries in the direction of 1w EMA50 trend, filtered by volume spike.
Weekly pivots provide robust support/resistance from institutional levels, EMA50 filters trend direction,
and volume spike confirms institutional participation. Designed for ~10-20 trades/year on 1d timeframe to minimize fee drag.
Works in both bull and bear markets via trend filter.
"""

name = "1d_WeeklyPivot_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Weekly pivot formulas
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    r2 = pivot + range_prev
    s2 = pivot - range_prev
    
    # Align weekly pivot levels to daily timeframe (wait for weekly bar to close)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get daily price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly pivot (needs 1 week), EMA50 (50 bars), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA50 (uptrend) AND price breaks above weekly R2 with volume spike
            if close[i] > ema_50_aligned[i] and high[i] > r2_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA50 (downtrend) AND price breaks below weekly S2 with volume spike
            elif close[i] < ema_50_aligned[i] and low[i] < s2_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR trend turns bearish
            if low[i] < pivot_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR trend turns bullish
            if high[i] > pivot_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals