#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout_VolumeConfirmation
Weekly pivot levels from 1w + 6h Donchian breakout with volume confirmation.
Long: price > weekly pivot R1 + Donchian(20) breakout + volume spike
Short: price < weekly pivot S1 + Donchian(20) breakdown + volume spike
Weekly pivot provides institutional reference; Donchian provides breakout signal.
Volume confirms institutional participation. Designed for 6h timeframe.
Target: 50-150 total trades over 4 years (12-37/year).
Works in bull/bear via breakout logic (works in trends) and pivot levels (mean reversion at extremes).
"""

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
    
    # === 1w Pivot Points (weekly high, low, close) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    # === 6h Donchian Channel (20-period) ===
    donchian_len = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(donchian_len - 1, len(high)):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
    
    # === 6h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # === Align 1w pivot levels to 6h timeframe ===
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = max(50, donchian_len, 20)
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: 1.5x average
        vol_spike = volume[i] > vol_ma_20[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price > weekly R1 + Donchian breakout + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > upper[i] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < weekly S1 + Donchian breakdown + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < lower[i] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price < weekly pivot OR Donchian breakdown
            if (close[i] < pivot_aligned[i] or 
                close[i] < lower[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > weekly pivot OR Donchian breakout
            if (close[i] > pivot_aligned[i] or 
                close[i] > upper[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0