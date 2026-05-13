#!/usr/bin/env python3
# 1d_KAMA_Trend_With_Volume_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction.
# Long when price > KAMA(14,2,30) with volume spike >1.5x20-day average.
# Short when price < KAMA with volume spike.
# Exit when price crosses back to KAMA (mean reversion).
# Uses daily timeframe for lower turnover and better trend capture in both bull and bear markets.

name = "1d_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
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

    # Get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate KAMA(14, 2, 30)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14:
            volatility[i] = np.nan
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[i-13:i+1])))

    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 14 or volatility[i] == 0:
            er[i] = 0
        else:
            er[i] = change[i] / volatility[i]

    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])

    # Align KAMA to 1d timeframe (same timeframe, no alignment needed, but for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)

    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if np.isnan(kama_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > KAMA with volume spike
            if close[i] > kama_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA with volume spike
            elif close[i] < kama_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses back to KAMA (mean reversion)
            if close[i] <= kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses back to KAMA
            if close[i] >= kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals