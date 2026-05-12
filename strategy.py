#!/usr/bin/env python3
# 1d_KAMA_Trend_With_Adaptive_Volume_Filter
# Hypothesis: KAMA adapts to market noise, capturing true trend in both bull and bear markets.
# Long when price > KAMA and volume > 20-day average * 1.5, short when price < KAMA and volume > 20-day average * 1.5.
# Uses 1d timeframe to avoid overtrading and focus on significant moves. Volume filter ensures trades occur
# during high conviction periods, reducing false signals. Target: 10-25 trades/year per symbol (40-100 total over 4 years).

name = "1d_KAMA_Trend_With_Adaptive_Volume_Filter"
timeframe = "1d"
leverage = 1.0

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

    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Parameters: fast=2, slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume filter: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(kama[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA with volume confirmation
            if close[i] > kama[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA with volume confirmation
            elif close[i] < kama[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals