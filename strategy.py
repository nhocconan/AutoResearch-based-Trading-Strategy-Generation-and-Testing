#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Volume
# Hypothesis: Combine weekly pivot levels with Donchian breakout and volume confirmation.
# Go long when price breaks above Donchian(20) high AND weekly pivot R1 level with volume > 1.5x average.
# Go short when price breaks below Donchian(20) low AND weekly pivot S1 level with volume > 1.5x average.
# Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts.
# Volume confirmation ensures breakouts have conviction. Designed for low frequency (15-35 trades/year)
# to avoid fee drag. Works in both bull (breakouts continue) and bear (breakdowns continue) markets.

name = "6h_WeeklyPivot_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """
    Calculate weekly pivot points: P, R1, R2, S1, S2.
    Uses typical price: (H + L + C) / 3
    Returns pivot, r1, r2, s1, s2 arrays.
    """
    typical_price = (high + low + close) / 3.0
    pivot = typical_price
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points
    pivot, r1, r2, s1, s2 = calculate_weekly_pivot(high_weekly, low_weekly, close_weekly)
    
    # Align weekly pivots to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= lookback - 1:
            highest_high[i] = np.max(high[i - lookback + 1:i + 1])
            lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i >= lookback - 1:
            avg_volume[i] = np.mean(volume[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback - 1
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x average volume
        volume_surge = volume[i] > 1.5 * avg_volume[i]
        
        if position == 0:
            # LONG: Break above Donchian high AND above weekly R1 with volume surge
            if close[i] > highest_high[i] and close[i] > r1_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low AND below weekly S1 with volume surge
            elif close[i] < lowest_low[i] and close[i] < s1_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls below Donchian low OR weekly pivot
            if close[i] < lowest_low[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above Donchian high OR weekly pivot
            if close[i] > highest_high[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals