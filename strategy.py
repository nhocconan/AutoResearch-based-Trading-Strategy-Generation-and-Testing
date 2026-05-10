#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_v2
Hypothesis: Use weekly pivot levels to filter Donchian(20) breakouts on 6h chart.
Long when price breaks above Donchian high and above weekly pivot resistance.
Short when price breaks below Donchian low and below weekly pivot support.
Weekly pivots calculated from prior week's OHLC using standard formula.
Works in bull/bear by following weekly pivot bias. Target: 15-30 trades/year.
"""

name = "6h_WeeklyPivot_Donchian_Breakout_v2"
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
    
    # Donchian channel (20-period) on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Weekly pivot levels (from prior week OHLC)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Standard pivot point: (H + L + C) / 3
    pivot = np.zeros(len(close_1w))
    # Resistance 1: (2 * P) - L
    r1 = np.zeros(len(close_1w))
    # Support 1: (2 * P) - H
    s1 = np.zeros(len(close_1w))
    # Resistance 2: P + (H - L)
    r2 = np.zeros(len(close_1w))
    # Support 2: P - (H - L)
    s2 = np.zeros(len(close_1w))
    
    for i in range(len(close_1w)):
        if i == 0:
            pivot[i] = (high_1w[0] + low_1w[0] + close_1w[0]) / 3
            r1[i] = (2 * pivot[i]) - low_1w[0]
            s1[i] = (2 * pivot[i]) - high_1w[0]
            r2[i] = pivot[i] + (high_1w[0] - low_1w[0])
            s2[i] = pivot[i] - (high_1w[0] - low_1w[0])
        else:
            pivot[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3
            r1[i] = (2 * pivot[i]) - low_1w[i]
            s1[i] = (2 * pivot[i]) - high_1w[i]
            r2[i] = pivot[i] + (high_1w[i] - low_1w[i])
            s2[i] = pivot[i] - (high_1w[i] - low_1w[i])
    
    # Align weekly pivots to 6h (no extra delay needed as pivots are known at week start)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Volume confirmation: current volume > 1.3x average volume (24-period = 4 days)
    vol_sma = np.full(n, np.nan)
    for i in range(24, n):
        vol_sma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24)  # Donchian + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(pivot_aligned[i]) or \
           np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 0:
            # Long: Break above Donchian high AND above weekly R1 (strong resistance)
            if close[i] > donchian_high[i] and close[i] > r1_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low AND below weekly S1 (strong support)
            elif close[i] < donchian_low[i] and close[i] < s1_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly pivot (trend weakness)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly pivot (trend weakness)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals