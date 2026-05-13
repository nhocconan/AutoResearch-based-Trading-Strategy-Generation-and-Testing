#!/usr/bin/env python3
# 1h_HMA_Trend_Filter + Volume + Session
# Hypothesis: HMA(21) on 4h defines trend, HMA(9) on 1h provides entry timing with volume confirmation.
# Session filter (08-20 UTC) reduces noise. Works in bull/bear via trend filter. Target: 20-60 trades/year.

name = "1h_HMA_Trend_Filter_Volume_Session"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # 4h HMA(21) for trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Hull MA: WMA(2*n, WMA(n, price)) - WMA(n, price)
    def wma(arr, n):
        if n <= 1:
            return arr.copy()
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights, mode='full')[:len(arr)] * weights / (weights.sum())
    def hull_ma(arr, n):
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        wma_half = wma(arr, half_n)
        wma_full = wma(arr, n)
        return wma(2 * wma_half - wma_full, sqrt_n)
    hma_21_4h = hull_ma(close_4h, 21)
    hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_4h)

    # 1h HMA(9) for entry
    def wma_series(arr, n):
        if n <= 1:
            return arr.copy()
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights, mode='full')[:len(arr)] * weights / (weights.sum())
    def hull_ma_series(arr, n):
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        wma_half = wma_series(arr, half_n)
        wma_full = wma_series(arr, n)
        return wma_series(2 * wma_half - wma_full, sqrt_n)
    hma_9_1h = hull_ma_series(close, 9)

    # Volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Session: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0

    for i in range(20, n):
        if (np.isnan(hma_21_4h_aligned[i]) or np.isnan(hma_9_1h[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h uptrend + 1h HMA rising + volume
            if (hma_21_4h_aligned[i] > hma_21_4h_aligned[i-1] and
                hma_9_1h[i] > hma_9_1h[i-1] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend + 1h HMA falling + volume
            elif (hma_21_4h_aligned[i] < hma_21_4h_aligned[i-1] and
                  hma_9_1h[i] < hma_9_1h[i-1] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h trend breaks or volume drops
            if (hma_21_4h_aligned[i] < hma_21_4h_aligned[i-1] or
                volume[i] < vol_avg_20[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend breaks or volume drops
            if (hma_21_4h_aligned[i] > hma_21_4h_aligned[i-1] or
                volume[i] < vol_avg_20[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals