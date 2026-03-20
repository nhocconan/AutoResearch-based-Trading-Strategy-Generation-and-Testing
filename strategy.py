#!/usr/bin/env python3
"""
EXPERIMENT #006 - KAMA (Kaufman Adaptive Moving Average) Trend Strategy
========================================================================
Hypothesis: KAMA adapts its smoothing constant based on market efficiency (noise vs trend).
During trending markets, KAMA moves faster to capture moves. During choppy markets, 
KAMA slows down to avoid whipsaws. This should outperform fixed EMA/HMA crossovers
which failed in experiments #002 and #004.

Key improvements:
- Adaptive smoothing based on Efficiency Ratio (ER)
- ER = |net change| / sum of absolute changes (0=noise, 1=perfect trend)
- Fast SC = 2/(2+1), Slow SC = 2/(2+30)
- 1h timeframe for more trade opportunities than 4h Supertrend
- Same conservative position sizing (0.35) to control DD
- Discrete signal levels to minimize churning costs
"""

import numpy as np
import pandas as pd

name = "kama_1h_v1"
timeframe = "1h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    n = len(close)
    
    if n < 50:
        return np.zeros(n)
    
    # KAMA Parameters
    er_period = 10  # Efficiency Ratio period
    fast_period = 2
    slow_period = 30
    
    # Calculate Efficiency Ratio (ER)
    # ER = |close - close[period]| / sum(|close[i] - close[i-1]|)
    er = np.zeros(n)
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = 0.0
        for j in range(1, er_period + 1):
            sum_changes += abs(close[i - j + 1] - close[i - j])
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant (SC)
    # SC = ER * (fast_sc - slow_sc) + slow_sc
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # Calculate KAMA
    kama = np.zeros(n)
    # Initialize KAMA with SMA of first er_period values
    kama[er_period] = np.mean(close[:er_period + 1])
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    # Generate signals based on price vs KAMA relationship
    # Long when price > KAMA and KAMA is rising
    # Short when price < KAMA and KAMA is falling
    signals = np.zeros(n)
    SIZE = 0.35  # 35% position size - critical for drawdown control
    
    # Add KAMA slope filter to reduce whipsaws
    kama_slope = np.zeros(n)
    slope_period = 5
    for i in range(slope_period, n):
        kama_slope[i] = kama[i] - kama[i - slope_period]
    
    for i in range(er_period + slope_period, n):
        if np.isnan(kama[i]) or np.isnan(kama_slope[i]):
            continue
        
        # Long signal: price above KAMA AND KAMA slope positive
        if close[i] > kama[i] and kama_slope[i] > 0:
            signals[i] = SIZE
        # Short signal: price below KAMA AND KAMA slope negative
        elif close[i] < kama[i] and kama_slope[i] < 0:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals