#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Donchian20_Breakout_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for pivot and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_donchian = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: price above/below Donchian midpoint
    donchian_mid = (upper_donchian + lower_donchian) / 2.0
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1w, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1w, lower_donchian)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Daily data for pivot points (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot points (standard)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.0 / 3.0)
    s1 = pivot - (range_1d * 1.0 / 3.0)
    r2 = pivot + range_1d
    s2 = pivot - range_1d
    r3 = pivot + (range_1d * 2.0)
    s3 = pivot - (range_1d * 2.0)
    
    # Align weekly trend and daily pivots to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(upper_donchian_aligned[i]) or
            np.isnan(lower_donchian_aligned[i]) or np.isnan(donchian_mid_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly upper Donchian AND above daily R3 (strong bullish)
            long_cond = (close[i] > upper_donchian_aligned[i] and 
                        close[i] > r3_aligned[i])
            
            # Short: Price breaks below weekly lower Donchian AND below daily S3 (strong bearish)
            short_cond = (close[i] < lower_donchian_aligned[i] and 
                         close[i] < s3_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price closes below weekly Donchian midpoint OR below daily S1
            if close[i] < donchian_mid_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price closes above weekly Donchian midpoint OR above daily R1
            if close[i] > donchian_mid_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals