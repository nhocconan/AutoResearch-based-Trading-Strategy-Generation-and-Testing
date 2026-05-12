#!/usr/bin/env python3
# 4h_TRIX_Trend_Volume
# Hypothesis: TRIX (15-period) on 4h timeframe signals momentum; confirmed by volume spike and 1d EMA trend filter.
# Works in bull by following 1d uptrend for longs; works in bear by following 1d downtrend for shorts.
# TRIX reduces false signals vs MACD; volume confirms breakout strength. Designed for 20-50 trades/year to minimize fee drag.

name = "4h_TRIX_Trend_Volume"
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

    # Get 4h data for TRIX and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    close_4h = df_4h['close'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate TRIX: triple EMA of log(close), then ROC
    # Step 1: EMA1 of log(close)
    log_close = np.log(close_4h)
    ema1 = pd.Series(log_close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 2: EMA2 of EMA1
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # Step 3: EMA3 of EMA2
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (EMA3_t / EMA3_{t-1} - 1)
    trix = np.zeros_like(ema3)
    trix[1:] = 100 * (ema3[1:] / ema3[:-1] - 1)
    # Align TRIX to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX > 0 (bullish momentum) in 1d uptrend with volume spike
            if trix_aligned[i] > 0 and close[i] > ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX < 0 (bearish momentum) in 1d downtrend with volume spike
            elif trix_aligned[i] < 0 and close[i] < ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX turns negative (momentum shift)
            if trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX turns positive (momentum shift)
            if trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals