#!/usr/bin/env python3
name = "4h_KAMA_Trend_200MA_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend and volume ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d KAMA for trend direction ===
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d, dtype=np.float64)
    for i in range(1, len(close_1d)):
        er[i] = change[i] / (volatility[i] + 1e-10) if volatility[i] > 0 else 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d, dtype=np.float64)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # === 1d 200-period MA for filter ===
    ma200_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(199, len(close_1d)):
        ma200_1d[i] = np.mean(close_1d[i-199:i+1])
    
    # === 1d volume spike detection ===
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=np.float64)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    
    # === Align 1d indicators to 4h timeframe ===
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    ma200_1d_aligned = align_htf_to_ltf(prices, df_1d, ma200_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(np.float64))
    
    # === 4h KAMA for entry timing ===
    # Efficiency ratio
    change_4h = np.abs(np.diff(close, prepend=close[0]))
    volatility_4h = np.abs(np.diff(close))
    er_4h = np.zeros_like(close, dtype=np.float64)
    for i in range(1, len(close)):
        er_4h[i] = change_4h[i] / (volatility_4h[i] + 1e-10) if volatility_4h[i] > 0 else 0
    # Smoothing constants
    sc_4h = (er_4h * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama_4h = np.zeros_like(close, dtype=np.float64)
    kama_4h[0] = close[0]
    for i in range(1, len(close)):
        kama_4h[i] = kama_4h[i-1] + sc_4h[i] * (close[i] - kama_4h[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 30, 200, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(ma200_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(kama_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above 1d KAMA AND above 1d MA200 AND volume spike
            if (close[i] > kama_1d_aligned[i] and 
                close[i] > ma200_1d_aligned[i] and
                volume_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price below 1d KAMA AND below 1d MA200 AND volume spike
            elif (close[i] < kama_1d_aligned[i] and 
                  close[i] < ma200_1d_aligned[i] and
                  volume_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below 4h KAMA or volume spike disappears
            if close[i] < kama_4h[i] or volume_spike_1d_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above 4h KAMA or volume spike disappears
            if close[i] > kama_4h[i] or volume_spike_1d_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals