#!/usr/bin/env python3
# 4h_Trix_30_ZeroCross_200EMA_Trend_Volume
# Hypothesis: TRIX(30) zero-cross filter with 200-period EMA trend filter and volume spike confirmation on 4h timeframe.
# TRIX is a momentum oscillator that filters out insignificant price movements. Zero-cross indicates trend change.
# Combined with 200 EMA for trend direction and volume spike for confirmation. Designed for low trade frequency.
# Works in bull/bear markets by following 200 EMA trend direction. Exit on TRIX zero-cross in opposite direction.
# Targets BTC/ETH with tight entry to avoid whipsaw and reduce trade frequency.

name = "4h_Trix_30_ZeroCross_200EMA_Trend_Volume"
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

    # Get 4h data for TRIX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    close_4h = df_4h['close'].values

    # Calculate TRIX: triple EMA of log returns
    # TRIX = EMA(EMA(EMA(log(close)), 15), 15), 15) * 100
    log_close = np.log(close_4h)
    ema1 = pd.Series(log_close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # First value has no previous

    # Align TRIX to lower timeframe
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)

    # Get 1d data for 200 EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Calculate volume spike threshold (2.0x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero in uptrend with volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema200_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero in downtrend with volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero (momentum shift)
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero (momentum shift)
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals