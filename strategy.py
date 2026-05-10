#!/usr/bin/env python3
"""
12h_WeeklyPivot_Donchian_Breakout_1wTrend_Volume
Hypothesis: 12h Donchian breakout in direction of 1w EMA50 trend, with weekly pivot support/resistance and volume confirmation.
Works in bull/bear by following weekly trend. Target: 15-35 trades/year.
"""

name = "12h_WeeklyPivot_Donchian_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Weekly pivot points (using weekly high, low, close)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_point = np.full(len(close_1w), np.nan)
    resistance_r1 = np.full(len(close_1w), np.nan)
    support_s1 = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i == 0:
            pivot_point[i] = close_1w[0]
            resistance_r1[i] = close_1w[0]
            support_s1[i] = close_1w[0]
        else:
            pivot_point[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
            resistance_r1[i] = 2 * pivot_point[i] - low_1w[i-1]
            support_s1[i] = 2 * pivot_point[i] - high_1w[i-1]
    pivot_point_aligned = align_htf_to_ltf(prices, df_1w, pivot_point)
    resistance_r1_aligned = align_htf_to_ltf(prices, df_1w, resistance_r1)
    support_s1_aligned = align_htf_to_ltf(prices, df_1w, support_s1)
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # EMA + Donchian + volume warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(pivot_point_aligned[i]) or np.isnan(resistance_r1_aligned[i]) or \
           np.isnan(support_s1_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above Donchian high, above weekly EMA50, and above weekly pivot
            if close[i] > donchian_high[i] and close[i] > ema_50_1w_aligned[i] and close[i] > pivot_point_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low, below weekly EMA50, and below weekly pivot
            elif close[i] < donchian_low[i] and close[i] < ema_50_1w_aligned[i] and close[i] < pivot_point_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly EMA50 or below weekly pivot
            if close[i] < ema_50_1w_aligned[i] or close[i] < pivot_point_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly EMA50 or above weekly pivot
            if close[i] > ema_50_1w_aligned[i] or close[i] > pivot_point_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals