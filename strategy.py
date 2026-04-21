#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_RSI_Momentum_V1
Hypothesis: Use 1d KAMA for trend direction and 4h RSI for momentum timing.
Long when 1d KAMA is rising and 4h RSI crosses above 50 from below.
Short when 1d KAMA is falling and 4h RSI crosses below 50 from above.
Exit when KAMA trend reverses.
Designed for 4h timeframe to capture intermediate trends with ~20-40 trades/year.
Works in bull markets by following uptrends and in bear markets by following downtrends.
KAMA adapts to volatility, reducing whipsaw in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)[:len(change)]  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i-10] * (close_1d[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.zeros_like(kama)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, -1)
    
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # Calculate 4h RSI
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])  # First 14-period average
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI crossing signals
    rsi_above = np.zeros_like(rsi, dtype=bool)
    rsi_below = np.zeros_like(rsi, dtype=bool)
    rsi_above[1:] = (rsi[1:] > 50) & (rsi[:-1] <= 50)
    rsi_below[1:] = (rsi[1:] < 50) & (rsi[:-1] >= 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(kama_dir_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising + RSI crosses above 50
            if kama_dir_aligned[i] == 1 and rsi_above[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling + RSI crosses below 50
            elif kama_dir_aligned[i] == -1 and rsi_below[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down
            if kama_dir_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up
            if kama_dir_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KAMA_Trend_RSI_Momentum_V1"
timeframe = "4h"
leverage = 1.0