#!/usr/bin/env python3
# 1d_KAMA_Trend_Filtered_With_Volume_Spike
# Hypothesis: KAMA adapts to market noise, providing a reliable trend signal with less whipsaw.
# In trending markets, price stays on one side of KAMA; in ranging markets, frequent crosses are filtered by volume spike and ADX.
# Long when price crosses above KAMA with volume spike and ADX > 25; short when price crosses below KAMA with volume spike and ADX > 25.
# Uses 1d for KAMA trend and ADX, and 1w for trend filter to avoid counter-trend trades in strong higher-timeframe trends.

name = "1d_KAMA_Trend_Filtered_With_Volume_Spike"
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

    # Calculate ER (Efficiency Ratio) for KAMA
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0

    # Smooth ER with smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate ADX
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0

    tr = np.maximum(
        high_1d - low_1d,
        np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    )
    tr[0] = high_1d[0] - low_1d[0]

    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing

    plus_di = 100 * np.where(atr != 0, np.convolve(plus_dm, np.ones(14)/14, mode='full')[:len(atr)] / atr, 0)
    minus_di = 100 * np.where(atr != 0, np.convolve(minus_dm, np.ones(14)/14, mode='full')[:len(atr)] / atr, 0)
    dx = 100 * np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.zeros_like(dx)
    for i in range(14, len(dx)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14 if i > 14 else np.mean(dx[14:28])

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma_20_1w = np.convolve(close_1w, np.ones(20)/20, mode='same')

    # Volume spike: current volume > 2 * 20-period average
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_spike = volume > 2 * vol_ma

    # Align indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(sma_20_1w_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price crosses above KAMA, volume spike, ADX > 25, and above weekly SMA
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and \
               vol_spike[i] and adx_aligned[i] > 25 and close[i] > sma_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA, volume spike, ADX > 25, and below weekly SMA
            elif close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and \
                 vol_spike[i] and adx_aligned[i] > 25 and close[i] < sma_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals