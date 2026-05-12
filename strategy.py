#!/usr/bin/env python3
"""
12h_KAMA_Direction_1dTrend_Volume
Hypothesis: KAMA adapts to market noise, providing a reliable trend signal. In trending markets (1d EMA34), KAMA direction signals entries with volume confirmation. Works in both bull and bear regimes by following the dominant trend. Designed for low trade frequency to avoid fee drag.
"""

name = "12h_KAMA_Direction_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate KAMA (12h timeframe)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    abs_change = np.sum(np.abs(np.diff(close, n=1)), axis=0) if len(close) >= 11 else np.zeros_like(close)
    # Handle array dimensions
    change = np.concatenate([np.full(10, np.nan), change]) if len(change) == len(close) - 10 else np.pad(change, (10, 0), constant_values=np.nan)
    abs_change = np.concatenate([np.full(1, np.nan), abs_change]) if len(abs_change) == len(close) - 1 else np.pad(abs_change, (1, 0), constant_values=np.nan)
    er = np.where(abs_change != 0, change / abs_change, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        if np.isnan(kama[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising + 1d uptrend + volume spike
            if kama[i] > kama[i-1] and close[i] > ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling + 1d downtrend + volume spike
            elif kama[i] < kama[i-1] and close[i] < ema34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling or 1d trend turns down
            if kama[i] < kama[i-1] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising or 1d trend turns up
            if kama[i] > kama[i-1] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals