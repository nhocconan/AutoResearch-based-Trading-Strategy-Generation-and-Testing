#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_Momentum
# Hypothesis: Use daily KAMA for trend direction, RSI for momentum confirmation, and volume spike for entry timing.
# Works in bull (KAMA rising, RSI > 50) and bear (KAMA falling, RSI < 50) by following the trend with momentum filters.
# Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_KAMA_Trend_RSI_Momentum"
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

    # KAMA calculation (10-period ER, 2/30 fast/slow SC)
    price_change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 10)))
    volatility = np.cumsum(price_change) - np.roll(np.cumsum(price_change), 10)
    volatility = np.where(volatility == 0, 1, volatility)
    er = direction / volatility
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_avg_20[i] = np.mean(volume[i-20:i])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN or invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (trend up) + RSI > 50 (bullish momentum) + volume spike
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (trend down) + RSI < 50 (bearish momentum) + volume spike
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI < 50
            if (kama[i] < kama[i-1] or rsi[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI > 50
            if (kama[i] > kama[i-1] or rsi[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals