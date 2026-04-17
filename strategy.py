#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d ATR(14) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.maximum(high_1d - low_1d,
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First value
    
    # ATR using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # === 1d EMA50 ===
    ema_50 = np.zeros(len(close_1d))
    alpha = 2 / (50 + 1)
    ema_50[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    
    # Align to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above EMA50 with low volatility
            if close[i] > ema_50_aligned[i] and atr_1d_aligned[i] < np.percentile(atr_1d[:i+1], 30):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below EMA50 with low volatility
            elif close[i] < ema_50_aligned[i] and atr_1d_aligned[i] < np.percentile(atr_1d[:i+1], 30):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: volatility expansion or mean reversion
        elif position == 1:
            # Exit long: volatility expansion or price below EMA50
            if atr_1d_aligned[i] > np.percentile(atr_1d[:i+1], 70) or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility expansion or price above EMA50
            if atr_1d_aligned[i] > np.percentile(atr_1d[:i+1], 70) or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA50_ATR_VolatilityFilter"
timeframe = "6h"
leverage = 1.0