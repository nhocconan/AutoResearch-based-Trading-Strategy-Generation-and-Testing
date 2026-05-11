#!/usr/bin/env python3
name = "6h_Donchian20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Weekly pivot points (from weekly data)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot and support/resistance levels
    pivot_weekly = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot_weekly - weekly_low
    s1 = 2 * pivot_weekly - weekly_high
    r2 = pivot_weekly + (weekly_high - weekly_low)
    s2 = pivot_weekly - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot_weekly - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot_weekly)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback - 1, 20)  # Ensure enough data for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_weekly_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly pivot R1 + volume confirmation
            if close[i] > highest_high[i] and close[i] > r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below weekly pivot S1 + volume confirmation
            elif close[i] < lowest_low[i] and close[i] < s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR below weekly pivot S1
            if close[i] < lowest_low[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR above weekly pivot R1
            if close[i] > highest_high[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals