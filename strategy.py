#!/usr/bin/env python3
# 4h_KAMA_Trend_Reversal
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) to detect trend direction with adaptive smoothing.
# Enter long when price crosses above KAMA with bullish momentum (positive price change over 3 periods).
# Enter short when price crosses below KAMA with bearish momentum (negative price change over 3 periods).
# Exit on opposite crossover. Uses 1d EMA50 as trend filter to avoid counter-trend trades in strong trends.
# Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull/bear markets via adaptive trend filter.

name = "4h_KAMA_Trend_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(close - np.roll(close, 10))
    change[0] = 0  # first value
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:10])  # placeholder for rolling sum
    # Recalculate volatility properly using rolling sum of absolute changes
    abs_changes = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.zeros(n)
    for i in range(10, n):
        volatility[i] = np.sum(abs_changes[i-9:i+1])
    # Avoid division by zero
    er = np.zeros(n)
    for i in range(10, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Momentum: price change over 3 periods
    mom = np.zeros(n)
    mom[0:3] = 0
    for i in range(3, n):
        mom[i] = close[i] - close[i-3]

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(mom[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price crosses above KAMA with bullish momentum and price > 1d EMA50
            if (close[i] > kama[i] and close[i-1] <= kama[i-1] and
                mom[i] > 0 and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA with bearish momentum and price < 1d EMA50
            elif (close[i] < kama[i] and close[i-1] >= kama[i-1] and
                  mom[i] < 0 and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals