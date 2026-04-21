#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_Momentum_V2
Hypothesis: Use KAMA direction on 1d to define trend, RSI for momentum confirmation, and avoid chop.
Long when KAMA rising + RSI > 50, short when KAMA falling + RSI < 50.
Exit when KAMA direction reverses.
Designed for 1d timeframe to capture multi-week moves with ~10-25 trades/year.
Works in bull markets by riding uptrends and in bear markets by shorting downtrends.
Avoids choppy markets with RSI near 50 filter.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Efficiency Ratio for KAMA (10-period)
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Pad the beginning with zeros for proper alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    for i in range(10, len(volatility)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat
    kama_dir = np.zeros_like(kama)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, np.where(kama[1:] < kama[:-1], -1, 0))
    
    # RSI (14-period)
    delta = np.diff(close)
    delta = np.concatenate([np.full(1, np.nan), delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    for i in range(14, len(close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:i+1])
            avg_loss[i] = np.mean(loss[1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after RSI and KAMA warmup
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Avoid choppy markets: RSI too close to 50 (indecision)
        if abs(rsi[i] - 50) < 5:  # RSI between 45 and 55
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA rising + RSI > 50
            if kama_dir[i] == 1 and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short conditions: KAMA falling + RSI < 50
            elif kama_dir[i] == -1 and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA direction turns down
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA direction turns up
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_Momentum_V2"
timeframe = "1d"
leverage = 1.0