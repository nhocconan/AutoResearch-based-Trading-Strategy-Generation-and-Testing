#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Uses weekly pivot levels to determine bias: long when price above weekly pivot, short when below.
Works in bull/bear by adapting to weekly structure. Target: 12-30 trades/year.
"""

name = "6h_WeeklyPivot_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0

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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    
    valid_idx = ~(np.isnan(high_1w) | np.isnan(low_1w) | np.isnan(close_1w))
    if np.any(valid_idx):
        pivot_1w[valid_idx] = (high_1w[valid_idx] + low_1w[valid_idx] + close_1w[valid_idx]) / 3.0
        r1_1w[valid_idx] = 2 * pivot_1w[valid_idx] - low_1w[valid_idx]
        s1_1w[valid_idx] = 2 * pivot_1w[valid_idx] - high_1w[valid_idx]
    
    # Align weekly pivot to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Donchian(20) on 6h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 2.0x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        if position == 0:
            # Long: Break above upper Donchian, above weekly pivot, with volume
            if close[i] > highest_high[i] and close[i] > pivot_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian, below weekly pivot, with volume
            elif close[i] < lowest_low[i] and close[i] < pivot_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly pivot
            if close[i] < pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly pivot
            if close[i] > pivot_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals