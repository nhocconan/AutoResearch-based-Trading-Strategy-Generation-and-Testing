#!/usr/bin/env python3
"""
4h_KAMA_Trend_Volume_Signal
Hypothesis: KAMA (Kaufman Adaptive Moving Average) captures trend with low lag in both trending and ranging markets.
On 4h timeframe, when price crosses above/below KAMA with volume confirmation, it signals trend continuation.
This adapts to market conditions, reducing whipsaw in sideways markets while capturing trends.
Volume confirmation ensures only significant moves are traded. Works in bull/bear markets as KAMA adapts to volatility.
"""

name = "4h_KAMA_Trend_Volume_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA Calculation (10-period ER, 2 and 30 for SC) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Correct volatility calculation: rolling sum of absolute changes
    volatility_rolling = pd.Series(np.abs(np.diff(close, prepend=close[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility_rolling > 0, change / volatility_rolling, 0)
    
    # Smoothing Constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # (fast - slow) * er + slow, then squared
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Volume Filter (1.5x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for KAMA
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(volume_ok[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals