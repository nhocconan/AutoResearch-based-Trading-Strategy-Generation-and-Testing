#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly high, low, close for pivot points
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot point and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Donchian channel (20-period) on 6h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        vol_ratio[i] = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high AND above weekly R1 AND volume confirmation
            long_cond = (close[i] > donchian_high[i]) and (close[i] > r1_aligned[i]) and (vol_ratio[i] > 1.5)
            
            # Short entry: price breaks below Donchian low AND below weekly S1 AND volume confirmation
            short_cond = (close[i] < donchian_low[i]) and (close[i] < s1_aligned[i]) and (vol_ratio[i] > 1.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Donchian low OR below weekly pivot
            if (close[i] < donchian_low[i]) or (close[i] < pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian high OR above weekly pivot
            if (close[i] > donchian_high[i]) or (close[i] > pivot_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points provide key support/resistance levels that institutions watch.
# Donchian(20) breakouts with volume confirmation capture institutional breakouts.
# Long when price breaks above Donchian high AND weekly R1 with volume > 1.5x average.
# Short when price breaks below Donchian low AND weekly S1 with volume > 1.5x average.
# Exits when price returns to weekly pivot or breaks Donchian in opposite direction.
# Works in trending markets (breakouts) and ranging markets (fade at extremes via pivot levels).
# Weekly timeframe ensures alignment with institutional weekly cycles.
# Volume filter ensures only significant breakouts are traded, reducing false signals.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.