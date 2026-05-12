#!/usr/bin/env python3
# 4h_KAMA_Triple_Confirmation
# Hypothesis: KAMA direction + RSI(14) extreme + volume spike (2x avg) for trend-following entries in 4h.
# Uses KAMA for adaptive trend direction, RSI for momentum confirmation, and volume to filter false breakouts.
# Exits when KAMA reverses. Designed for 20-30 trades/year with low turnover to minimize fee drag.
# Works in bull/bear markets by following adaptive trend; avoids whipsaw via volume confirmation.

name = "4h_KAMA_Triple_Confirmation"
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

    # Calculate KAMA (adaptive trend)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])).reshape(-1, 1), axis=1)  # placeholder, will fix
    # Correct ER calculation
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility_sum = np.sum(np.abs(np.diff(close[i-9:i+1])))
            er[i] = direction / volatility_sum if volatility_sum != 0 else 0
    # Avoid division by zero
    er = np.where(np.isnan(er), 0, er)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike: 2x 20-period average
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_sma20 * 2.0)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA up, RSI > 50, volume spike
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA down, RSI < 50, volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA reverses down
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA reverses up
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals