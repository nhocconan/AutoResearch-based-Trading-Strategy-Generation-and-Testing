#!/usr/bin/env python3
# 4h_KAMA_Trend_Volume_Filter
# Hypothesis: KAMA adapts to market noise, providing smooth trend signals. In trending markets, price stays above/below KAMA with sustained direction. Combining KAMA trend with volume confirmation filters out whipsaws. Works in bull/bear markets by following the trend via KAMA direction, with volume spikes adding conviction. Targets ~30 trades/year to minimize fee drag.

name = "4h_KAMA_Trend_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio = |net change| / sum(|changes|)
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA = prevKAMA + SC * (price - prevKAMA)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    delta = np.abs(np.subtract(close, np.roll(close, 1)))
    delta[0] = 0

    er = np.abs(np.subtract(close, np.roll(close, 10))) / \
         (np.abs(np.diff(close, n=10)) + 1e-10)  # avoid div by zero
    er[0:10] = 0  # first 10 periods invalid

    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2
    sc[0:10] = 0

    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Alternative vectorized approach using pandas for clarity and performance
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=1).sum()
    net_change = close_series.diff(periods=10).abs()
    er = net_change / (volatility + 1e-10)
    er = er.fillna(0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2  # 2/(2+2)=0.6667, 2/(30+2)=0.0645
    kama_series = close_series.copy()
    for i in range(1, len(close_series)):
        kama_series.iloc[i] = kama_series.iloc[i-1] + sc.iloc[i] * (close_series.iloc[i] - kama_series.iloc[i-1])
    kama = kama_series.values

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
            # LONG: price > KAMA + volume confirmation
            if close[i] > kama[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA + volume confirmation
            elif close[i] < kama[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals