#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_1d_RSI_Filter
Hypothesis: Use daily KAMA to determine trend direction and daily RSI for overbought/oversold conditions.
Enter long when KAMA turns up (bullish) and RSI < 40 (oversold pullback).
Enter short when KAMA turns down (bearish) and RSI > 60 (overbought bounce).
Exit when KAMA reverses direction. Uses 1-day timeframe to limit trades and avoid fee drag.
Works in bull markets via pullbacks to rising KAMA and bear via bounces to falling KAMA.
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on close
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 10)))
    volatility = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    # Pad volatility to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = np.power(er * (0.6665 - 0.0645) + 0.0645, 2)
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after KAMA and RSI warmup
    start_idx = 14
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA turning up (current > previous) AND RSI oversold (< 40)
            if kama[i] > kama[i-1] and rsi[i] < 40:
                signals[i] = size
                position = 1
            # Short: KAMA turning down (current < previous) AND RSI overbought (> 60)
            elif kama[i] < kama[i-1] and rsi[i] > 60:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA turns up
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_Trend_With_1d_RSI_Filter"
timeframe = "1d"
leverage = 1.0