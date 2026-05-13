#!/usr/bin/env python3
# 4h_KAMA_Trend_Volume_Pullback
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 4h to determine trend,
# combined with volume confirmation and pullback to KAMA for entry.
# In trending markets, price tends to pull back to the KAMA before continuing.
# Volume spike confirms institutional interest. Works in both bull and bear by
# following the trend direction. Target: 25-35 trades/year on 4h to minimize fee drag.

name = "4h_KAMA_Trend_Volume_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate KAMA (4h) - trend indicator
    # Parameters: ER period=10, Fast SC=2, Slow SC=30
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0) if hasattr(np, 'diff') else np.sum(np.abs(np.diff(close)), axis=0)
    # Manual calculation for volatility sum over 10 periods
    volatility_sum = np.zeros_like(close)
    for i in range(n):
        if i >= 10:
            volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
        else:
            volatility_sum[i] = 1.0  # avoid division by zero
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    sc = (er * (2/2 - 30/30) + 30/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = np.zeros_like(volume)
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values

    # Pullback condition: price within 0.5% of KAMA (avoid chasing)
    pullback_threshold = 0.005  # 0.5%

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
            # LONG: Price above KAMA (uptrend) + pullback to KAMA + volume spike
            if (close[i] > kama[i] and
                abs(close[i] - kama[i]) / kama[i] <= pullback_threshold and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + pullback to KAMA + volume spike
            elif (close[i] < kama[i] and
                  abs(close[i] - kama[i]) / kama[i] <= pullback_threshold and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA (trend change)
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA (trend change)
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals