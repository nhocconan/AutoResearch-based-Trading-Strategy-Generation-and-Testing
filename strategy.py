#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Filter_Trend
Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 4h to determine trend direction, combined with RSI(14) for momentum confirmation and volume spike filter. Enter long when KAMA slopes up and RSI > 55, short when KAMA slopes down and RSI < 45. Avoids whipsaws in sideways markets by requiring both trend and momentum alignment. Works in bull/bear by following adaptive trend.
"""

name = "4h_KAMA_Direction_RSI_Filter_Trend"
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
    volume = prices['volume'].values

    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    er = np.concatenate([np.full(er_length, np.nan), er])

    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    sc = np.nan_to_num(sc, nan=0.0)

    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length] = close[er_length]
    for i in range(er_length + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after KAMA/RSI warmup
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising (trend up) + RSI > 55 (bullish momentum) + volume spike
            if kama[i] > kama[i-1] and rsi[i] > 55 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (trend down) + RSI < 45 (bearish momentum) + volume spike
            elif kama[i] < kama[i-1] and rsi[i] < 45 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down (trend change)
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up (trend change)
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals