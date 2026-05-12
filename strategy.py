#!/usr/bin/env python3
name = "4h_KAMA_1dTrend_VolumeSpike"
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
    
    # === 1d Data for trend and volume ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d EMA34 for trend ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 1d Volume Spike Detection ===
    vol_ma_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # === 4h KAMA (Adaptive Moving Average) ===
    # KAMA parameters: ER = efficiency ratio, SC = smoothing constant
    change_4h = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility_4h = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    
    # Pad arrays for alignment
    change_4h = np.concatenate([np.full(9, np.nan), change_4h])
    volatility_4h = np.concatenate([np.full(9, np.nan), volatility_4h])
    
    # Avoid division by zero
    er_4h = np.divide(change_4h, volatility_4h, out=np.full_like(change_4h, np.nan), where=volatility_4h!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc_4h = (er_4h * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama_4h = np.full_like(close, np.nan)
    kama_4h[9] = close[9]  # Start with first available close
    for i in range(10, n):
        if not np.isnan(sc_4h[i]):
            kama_4h[i] = kama_4h[i-1] + sc_4h[i] * (close[i] - kama_4h[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 10, 10)  # EMA34, KAMA warmup, volume
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(kama_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA + 1d uptrend + volume spike
            if (close[i] > kama_4h[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and  # 1d EMA rising
                volume_spike_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + 1d downtrend + volume spike
            elif (close[i] < kama_4h[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and  # 1d EMA falling
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below KAMA or 1d trend reversal
            if close[i] < kama_4h[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above KAMA or 1d trend reversal
            if close[i] > kama_4h[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals