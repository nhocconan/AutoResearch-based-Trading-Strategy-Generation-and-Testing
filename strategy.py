# 4h_LongOnly_KAMA_RSI_Trend_v1
#!/usr/bin/env python3
"""
Long-only KAMA-based trend following strategy for 4h timeframe.
- Uses KAMA (Kaufman Adaptive Moving Average) to capture trend direction
- Filters with RSI (14) to avoid overbought/oversold extremes
- Only takes long positions (no shorts) to reduce whipsaw in volatile markets
- Exits when trend weakens or RSI indicates exhaustion
- Designed for ~20-40 trades/year to minimize fee drag
"""

import numpy as np
import pandas as pd

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    if n < er_length:
        return np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    change = np.abs(close[er_length:] - close[:-er_length])
    volatility = np.sum(np.abs(np.diff(close[:n-er_length+1])), axis=0)
    er = np.zeros(n)
    er[er_length:] = change / np.maximum(volatility, 1e-10)
    
    # Calculate smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_length-1] = close[er_length-1]
    for i in range(er_length, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close), np.nan)
    avg_loss = np.full(len(close), np.nan)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Enter long: price above KAMA and RSI not overbought
            if close[i] > kama[i] and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit long: price below KAMA or RSI overbought
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "4h_LongOnly_KAMA_RSI_Trend_v1"
timeframe = "4h"
leverage = 1.0