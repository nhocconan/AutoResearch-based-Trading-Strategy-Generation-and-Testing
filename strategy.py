#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Confirmation
Trend-following strategy using Kaufman Adaptive Moving Average (KAMA) on 4h timeframe.
Long when price > KAMA(14,2,30) and volume > 1.5x average volume.
Short when price < KAMA(14,2,30) and volume > 1.5x average volume.
Uses 12h trend filter (EMA50) to avoid counter-trend trades.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    kama_period = 14
    fast_ema = 2
    slow_ema = 30
    
    # Calculate directional change
    change = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate volatility
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])))
    
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1, volatility)
    
    # Efficiency Ratio
    er = np.zeros(n)
    er[0] = 0
    for i in range(1, n):
        er[i] = change[i] / volatility if volatility != 0 else 0
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Average volume for confirmation
    vol_ma_period = 20
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= vol_ma_period:
            vol_sum -= volume[i - vol_ma_period]
        if i >= vol_ma_period - 1:
            vol_ma[i] = vol_sum / vol_ma_period
        else:
            vol_ma[i] = 0
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.zeros(len(close_12h))
    ema_12h[0] = close_12h[0]
    alpha = 2 / (ema_period + 1)
    for i in range(1, len(close_12h)):
        ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    
    # Align 12h EMA50 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need KAMA, volume MA, and EMA12h
    start_idx = max(kama_period, vol_ma_period - 1, ema_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if vol_ma[i] == 0 or np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        ema12h_val = ema_12h_aligned[i]
        
        if position == 0:
            # Long: price > KAMA, volume confirmation, and above 12h EMA
            if price > kama_val and vol_ratio > 1.5 and price > ema12h_val:
                signals[i] = size
                position = 1
            # Short: price < KAMA, volume confirmation, and below 12h EMA
            elif price < kama_val and vol_ratio > 1.5 and price < ema12h_val:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or trend fails
            if price < kama_val or price < ema12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above KAMA or trend fails
            if price > kama_val or price > ema12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0