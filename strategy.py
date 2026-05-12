#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Confirmation
Hypothesis: On 4h timeframe, use KAMA to capture adaptive trends, with volume >1.5x average to confirm momentum, targeting 30-50 trades per year. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing strong trends in both bull and bear markets. Volume filter ensures only high-confidence breakouts are taken, reducing false signals and keeping trade frequency low to overcome fee drag.
"""

name = "4h_KAMA_Trend_With_Volume_Confirmation"
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

    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER period = 10, Fast EMA = 2, Slow EMA = 30
    er_period = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # placeholder, will fix below
    # Recalculate volatility properly: sum of absolute changes over er_period
    volatility = np.zeros_like(close)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))

    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    er[0] = 0  # first value

    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(kama[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above KAMA with volume confirmation
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA with volume confirmation
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals