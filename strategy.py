#!/usr/bin/env python3
# 4h_VolumeSpike_KAMA_Reversal
# Hypothesis: KAMA trend direction combined with volume spikes captures reversal moves
# in both bull and bear markets. Uses 1d trend filter to avoid counter-trend trades.
# Low-frequency design (target 20-30 trades/year) to minimize fee drag.

name = "4h_VolumeSpike_KAMA_Reversal"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=er_period))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > er_period else np.zeros_like(close_1d)
    # Vectorized volatility calculation
    volatility = np.full_like(close_1d, np.nan)
    for i in range(er_period, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-er_period+1:i+1])))
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[er_period-1] = close_1d[er_period-1]
    for i in range(er_period, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume confirmation (20-period average on 4h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30) + 5
    
    for i in range(start_idx, n):
        if np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.5x average (strict for low frequency)
        volume_spike = volume[i] > 2.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price crosses above KAMA with volume spike
            if close[i] > kama_1d_aligned[i] and close[i-1] <= kama_1d_aligned[i-1] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume spike
            elif close[i] < kama_1d_aligned[i] and close[i-1] >= kama_1d_aligned[i-1] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama_1d_aligned[i] and close[i-1] >= kama_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama_1d_aligned[i] and close[i-1] <= kama_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals