#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Donchian_Breakout
Hypothesis: Combines weekly pivot point direction (from prior week) with 6h Donchian breakouts and volume confirmation. Trades only in direction of weekly pivot bias, filtering out counter-trend moves. Designed to work in bull/bear markets by using weekly structure as trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's HLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    pivot = np.full(len(df_weekly), np.nan)
    r1 = np.full(len(df_weekly), np.nan)
    s1 = np.full(len(df_weekly), np.nan)
    r2 = np.full(len(df_weekly), np.nan)
    s2 = np.full(len(df_weekly), np.nan)
    
    for i in range(1, len(df_weekly)):
        if np.isnan(weekly_high[i-1]) or np.isnan(weekly_low[i-1]) or np.isnan(weekly_close[i-1]):
            continue
        pivot[i] = (weekly_high[i-1] + weekly_low[i-1] + weekly_close[i-1]) / 3.0
        r1[i] = 2 * pivot[i] - weekly_low[i-1]
        s1[i] = 2 * pivot[i] - weekly_high[i-1]
        r2[i] = pivot[i] + (weekly_high[i-1] - weekly_low[i-1])
        s2[i] = pivot[i] - (weekly_high[i-1] - weekly_low[i-1])
    
    # Align weekly pivots to 6h (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 30  # Warmup for Donchian and weekly pivot
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and above weekly pivot/R1
            if close[i] > donchian_high[i] and vol_spike[i] and close[i] > pivot_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below Donchian low with volume spike and below weekly pivot/S1
            elif close[i] < donchian_low[i] and vol_spike[i] and close[i] < pivot_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 2 bars hold, then exit on trend reversal or volatility drop
            if bars_since_entry >= 2:
                if close[i] < pivot_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 2 bars hold, then exit on trend reversal or volatility drop
            if bars_since_entry >= 2:
                if close[i] > pivot_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "6h_Weekly_Pivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0