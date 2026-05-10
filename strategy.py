#!/usr/bin/env python3
# 6h_Camarilla_Pivot_WeeklyTrend_Filter
# Hypothesis: Camarilla pivot levels from daily timeframe provide precise entry/exit zones,
# while weekly trend filter (price above/below weekly EMA20) ensures trades align with
# higher timeframe momentum. Works in bull markets via buying dips at S1/S2 in uptrend
# and selling rallies at R1/R2 in downtrend. Low trade frequency due to dual-timeframe
# confirmation and strict pivot-based entries. Target: 20-40 trades/year.

name = "6h_Camarilla_Pivot_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Pivot point
    pivot = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    r4 = close + range_val * 1.500
    r3 = close + range_val * 1.250
    r2 = close + range_val * 1.166
    r1 = close + range_val * 1.083
    s1 = close - range_val * 1.083
    s2 = close - range_val * 1.166
    s3 = close - range_val * 1.250
    s4 = close - range_val * 1.500
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    _, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # Align Camarilla levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Get 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily pivot (1) + weekly EMA (20) + volume EMA (20)
    start_idx = max(1, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or
            np.isnan(s1_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above S1 in uptrend (price > weekly EMA20)
            if close[i] > s1_6h[i] and close[i-1] <= s1_6h[i-1] and close[i] > weekly_ema20_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 in downtrend (price < weekly EMA20)
            elif close[i] < r1_6h[i] and close[i-1] >= r1_6h[i-1] and close[i] < weekly_ema20_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S2 OR trend reverses
            if close[i] < s2_6h[i] and close[i-1] >= s2_6h[i-1] or close[i] < weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R2 OR trend reverses
            if close[i] > r2_6h[i] and close[i-1] <= r2_6h[i-1] or close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals