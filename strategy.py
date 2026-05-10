#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian_Breakout_TrendFilter
Hypothesis: Weekly pivot levels act as strong support/resistance. Donchian(20) breakout in direction of weekly pivot (above/below pivot) with volume confirmation. Uses 1d EMA50 for trend filter to avoid counter-trend trades. Works in bull/bear by following higher timeframe structure.
Target: 15-25 trades/year per symbol.
"""

name = "6h_WeeklyPivot_Donchian_Breakout_TrendFilter"
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
    
    # Weekly high/low for pivot calculation (use weekly bar)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point: (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 1.5x average volume
    vol_sma = np.full(n, np.nan)
    for i in range(10, n):  # shorter MA for volume
        vol_sma[i] = np.mean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 10)
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Donchian breakout above weekly pivot and above EMA50_1d
            if high[i] > donchian_high[i] and close[i] > pivot_1w_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below weekly pivot and below EMA50_1d
            elif low[i] < donchian_low[i] and close[i] < pivot_1w_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly pivot or below EMA50_1d
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly pivot or above EMA50_1d
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals