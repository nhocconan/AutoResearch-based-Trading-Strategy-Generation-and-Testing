#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Momentum_VolumeFilter
Hypothesis: KAMA adapts to market noise, capturing trend direction efficiently. Combined with RSI momentum and volume confirmation on daily timeframe, it captures strong moves while avoiding chop. Works in bull/bear by following KAMA trend direction with momentum confirmation.
"""

name = "1d_KAMA_Trend_With_RSI_Momentum_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # KAMA (Kaufman Adaptive Moving Average) - ER=10, Fast=2, Slow=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros_like(change)
    er[volatility != 0] = change[volatility != 0] / volatility[volatility != 0]
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Volume confirmation: >1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after KAMA/RSI warmup
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_confirm[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + RSI > 50 (momentum) + volume confirmation
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + RSI < 50 (momentum) + volume confirmation
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals