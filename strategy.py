#!/usr/bin/env python3
"""
1d_KAMA_Trend_v1
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) to capture primary trend on daily timeframe.
Trend-following strategy designed for low trade frequency (~10-20 trades/year) by using a single 
adaptive moving average crossover. Performs well in both bull and bear markets by adapting to 
volatility changes - faster in trending markets, slower in choppy conditions.
"""

name = "1d_KAMA_Trend_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # --- KAMA (10-period efficiency ratio) ---
    # Calculate 10-period change
    change = np.abs(np.subtract(close[10:], close[:-10]))
    # Calculate sum of absolute daily changes
    abs_diff = np.abs(np.diff(close))
    # Create array for efficiency ratio
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if np.sum(abs_diff[i-9:i+1]) > 0:
            er[i] = change[i-10] / np.sum(abs_diff[i-9:i+1])
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # KAMA trend following
        if position == 0:
            if close[i] > kama[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < kama[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals