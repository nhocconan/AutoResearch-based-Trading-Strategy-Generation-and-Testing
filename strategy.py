#!/usr/bin/env python3
"""
1d_1w_WeeklyPivot_Donchian_Breakout_Volume
Hypothesis: On 1d timeframe, trade breakout of weekly pivot levels (R1/S1) in direction of weekly trend (10-week EMA) with volume confirmation. Works in bull/bear by following weekly trend. Target: 10-25 trades/year.
"""

name = "1d_1w_WeeklyPivot_Donchian_Breakout_Volume"
timeframe = "1d"
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
    
    # Weekly trend: 10-period EMA on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_10_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 10:
        ema_10_1w[9] = np.mean(close_1w[:10])
        alpha = 2 / (10 + 1)
        for i in range(10, len(close_1w)):
            ema_10_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_10_1w[i-1]
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Weekly pivot points (R1, S1)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pivot_1w = np.full(len(close_1w), np.nan)
    r1_1w = np.full(len(close_1w), np.nan)
    s1_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        pivot_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        r1_1w[i] = pivot_1w[i] + (high_1w[i] - low_1w[i]) * 1.0833
        s1_1w[i] = pivot_1w[i] - (high_1w[i] - low_1w[i]) * 1.0833
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10)  # volume and EMA warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_10_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R1 and above weekly EMA10
            if close[i] > r1_1w_aligned[i] and close[i] > ema_10_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 and below weekly EMA10
            elif close[i] < s1_1w_aligned[i] and close[i] < ema_10_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below weekly EMA10
            if close[i] < ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above weekly EMA10
            if close[i] > ema_10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals