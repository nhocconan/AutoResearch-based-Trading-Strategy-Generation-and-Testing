#!/usr/bin/env python3
"""
1d_PhaseAccumulation_Momentum
Hypothesis: Use daily Ehlers' Phase Accumulation indicator to detect cyclic momentum turns.
Long when phase crosses above 30° (momentum building), short when crosses below -30°.
Add volume confirmation and weekly trend filter to avoid false signals in chop.
Designed for low frequency: ~10-20 trades/year per symbol to minimize fee drag.
Works in bull/bear by capturing momentum shifts at cycle extremes.
"""

name = "1d_PhaseAccumulation_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from math import atan, sqrt
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data (already native)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Simple weekly trend: price above/below 20-week EMA
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    # Calculate Ehlers' Phase Accumulation indicator
    # Using 1-bar momentum (close - close[1]) as the input
    momentum = np.zeros(n)
    momentum[0] = 0
    for i in range(1, n):
        momentum[i] = close[i] - close[i-1]
    
    # Smooth momentum with a 4-bar SuperSmoother (alpha=0.1)
    smoothed = np.zeros(n)
    smoothed[0] = momentum[0]
    alpha = 0.1
    for i in range(1, n):
        smoothed[i] = alpha * momentum[i] + (1 - alpha) * smoothed[i-1]
    
    # Calculate phase using arctan of quadrature components
    # Simplified: use smoothed momentum and its derivative
    delta = np.zeros(n)
    delta[0] = 0
    for i in range(1, n):
        delta[i] = smoothed[i] - smoothed[i-1]
    
    # Avoid division by zero
    denom = smoothed + 1e-10
    phase = np.degrees(atan(delta / denom))
    
    # Weekly trend alignment
    weekly_uptrend = close_1w > ema20_1w
    weekly_downtrend = close_1w < ema20_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma20 = np.full(n, np.nan)
    if n >= 20:
        vol_ma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    volume_confirm = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(vol_ma20[i]) or np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: phase crosses above 30°, weekly uptrend, volume confirmation
            if phase[i] > 30 and phase[i-1] <= 30 and weekly_uptrend_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: phase crosses below -30°, weekly downtrend, volume confirmation
            elif phase[i] < -30 and phase[i-1] >= -30 and weekly_downtrend_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: phase crosses below 0 or weekly trend turns down
            if phase[i] < 0 or not weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: phase crosses above 0 or weekly trend turns up
            if phase[i] > 0 or not weekly_downtrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals