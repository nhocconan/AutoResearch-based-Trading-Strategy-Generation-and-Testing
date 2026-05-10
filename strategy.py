#!/usr/bin/env python3
"""
1d_WeeklyPivot_Pullback_With_Volume
Hypothesis: In trending markets (above/below weekly EMA50), enter on pullbacks to weekly pivot (P) with volume confirmation.
In bull trend: buy when price pulls back to pivot P after touching R1.
In bear trend: sell when price pulls back to pivot P after touching S1.
Uses weekly pivot levels as dynamic support/resistance with trend filter.
Designed for 15-25 trades/year, avoids overtrading while capturing trend continuation.
Works in both bull (buy pullbacks) and bear (sell pullbacks) markets.
"""

name = "1d_WeeklyPivot_Pullback_With_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    high_prev = df_weekly['high'].shift(1).values
    low_prev = df_weekly['low'].shift(1).values
    close_prev = df_weekly['close'].shift(1).values
    
    # Standard pivot: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot_p = (high_prev + low_prev + close_prev) / 3
    pivot_r1 = 2 * pivot_p - low_prev
    pivot_s1 = 2 * pivot_p - high_prev
    
    # Align weekly pivot levels to 1d timeframe (wait for weekly bar to close)
    pivot_p_aligned = align_htf_to_ltf(prices, df_weekly, pivot_p)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_weekly, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_weekly, pivot_s1)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_weekly, ema_50)
    
    # Get 1d price and volume
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
            # Long: uptrend AND pulled back to pivot P after touching R1 (higher low at P)
            if (close[i] > ema_50_aligned[i] and  # uptrend filter
                low[i] <= pivot_p_aligned[i] * 1.001 and  # touched or slightly below P
                high[i-1] > pivot_r1_aligned[i-1] and  # previously touched R1
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND pulled back to pivot P after touching S1 (lower high at P)
            elif (close[i] < ema_50_aligned[i] and  # downtrend filter
                  high[i] >= pivot_p_aligned[i] * 0.999 and  # touched or slightly above P
                  low[i-1] < pivot_s1_aligned[i-1] and  # previously touched S1
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below pivot S1 OR trend turns bearish
            if low[i] < pivot_s1_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above pivot R1 OR trend turns bullish
            if high[i] > pivot_r1_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals