#!/usr/bin/env python3
name = "1d_WeeklyHullTrend_WeeklyVolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hull_moving_average(arr, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / (window * (window + 1) / 2)
    
    wma_half = np.full(n, np.nan)
    wma_full = np.full(n, np.nan)
    
    for i in range(half, n):
        wma_half[i] = wma(arr[i-half+1:i+1], half)[-1] if i-half+1 >= 0 else np.nan
    for i in range(period, n):
        wma_full[i] = wma(arr[i-period+1:i+1], period)[-1] if i-period+1 >= 0 else np.nan
    
    raw = 2 * wma_half - wma_full
    hull = np.full(n, np.nan)
    
    for i in range(sqrt_n-1, n):
        if not np.isnan(raw[i]):
            hull[i] = wma(raw[i-sqrt_n+1:i+1], sqrt_n)[-1] if i-sqrt_n+1 >= 0 else np.nan
    
    return hull

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for trend and volume ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # === Weekly Hull Moving Average (21) for trend ===
    hull_21_1w = hull_moving_average(close_1w, 21)
    hull_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hull_21_1w)
    
    # === Weekly Volume Spike Detection ===
    vol_ma_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_spike_1w = volume_1w > (vol_ma_1w * 2.0)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hull_21_1w_aligned[i]) or 
            np.isnan(volume_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above weekly Hull MA + weekly volume spike
            if (close[i] > hull_21_1w_aligned[i] and 
                volume_spike_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly Hull MA + weekly volume spike
            elif (close[i] < hull_21_1w_aligned[i] and 
                  volume_spike_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below weekly Hull MA
            if close[i] < hull_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above weekly Hull MA
            if close[i] > hull_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals