#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation
# Uses weekly pivot levels to establish long-term bias (above/below weekly pivot) and
# 6h Donchian breakouts for entry timing. Volume confirms breakout strength.
# Works in bull markets (buy above pivot + breakout up) and bear markets (sell below pivot + breakout down).
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data (primary timeframe) for Donchian and price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1w data for weekly pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channel (20-period) on 6h
    dc_high_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    dc_low_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align all indicators to 6h timeframe
    dc_high_aligned = align_htf_to_ltf(prices, df_6h, dc_high_20)
    dc_low_aligned = align_htf_to_ltf(prices, df_6h, dc_low_20)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_high_aligned[i]) or np.isnan(dc_low_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
            continue
        
        # Long entry: price above weekly pivot + breaks above Donchian high + volume confirmation
        if (close[i] > pivot_aligned[i] and
            close[i] > dc_high_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price below weekly pivot + breaks below Donchian low + volume confirmation
        elif (close[i] < pivot_aligned[i] and
              close[i] < dc_low_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse Donchian breakout or price crosses weekly pivot in opposite direction
        elif position == 1 and (close[i] < dc_low_aligned[i] or close[i] < pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > dc_high_aligned[i] or close[i] > pivot_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0