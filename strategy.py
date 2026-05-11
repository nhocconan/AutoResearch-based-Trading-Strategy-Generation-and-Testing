#!/usr/bin/env python3
name = "4h_KAMA_Trend_Volume_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 12h close (ER=10, fast=2, slow=30)
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)
    # Manual calculation for efficiency
    er = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i < 10:
            er[i] = 0
        else:
            direction = np.abs(close_12h[i] - close_12h[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
            er[i] = direction / (volatility_sum + 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # 12h volume filter: current volume > 1.3x 20-period average
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter = vol_12h > (vol_ma_12h * 1.3)
    
    # Align to 4h
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    volume_filter_aligned = align_htf_to_ltf(prices, df_12h, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if np.isnan(kama_aligned[i]) or np.isnan(volume_filter_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA and volume filter
            if close[i] > kama_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and volume filter
            elif close[i] < kama_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals