#!/usr/bin/env python3
# 4H_1D_EMA_CROSSOVER_VOLUME_TREND
# Hypothesis: Daily EMA crossover (21/55) combined with 4h volume and trend alignment provides high-probability entries.
# EMA21 > EMA55 indicates uptrend on daily; EMA21 < EMA55 indicates downtrend. Entry requires price to be above/below both EMA on 4h,
# volume > 1.5x 20-period average, and 4h close aligned with daily trend. Exit on EMA crossover reversal.
# Designed for fewer trades (target 20-40/year) with strong trend capture in both bull and bear markets.

name = "4H_1D_EMA_CROSSOVER_VOLUME_TREND"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # 4h EMA21 and EMA55
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55 = pd.Series(close).ewm(span=55, adjust=False, min_periods=55).mean().values

    # Daily EMA21 and EMA55 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_1d = pd.Series(df_1d['close'].values).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    ema55_1d_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(55, n):
        # Skip if any required value is NaN
        if (np.isnan(ema21[i]) or np.isnan(ema55[i]) or 
            np.isnan(ema21_1d_aligned[i]) or np.isnan(ema55_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Daily EMA21 > EMA55 (uptrend) + 4h close > both EMA + volume spike
            if (ema21_1d_aligned[i] > ema55_1d_aligned[i] and 
                close[i] > ema21[i] and close[i] > ema55[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Daily EMA21 < EMA55 (downtrend) + 4h close < both EMA + volume spike
            elif (ema21_1d_aligned[i] < ema55_1d_aligned[i] and 
                  close[i] < ema21[i] and close[i] < ema55[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Daily EMA21 < EMA55 (trend reversal) or 4h close < EMA21
            if (ema21_1d_aligned[i] < ema55_1d_aligned[i] or close[i] < ema21[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Daily EMA21 > EMA55 (trend reversal) or 4h close > EMA21
            if (ema21_1d_aligned[i] > ema55_1d_aligned[i] or close[i] > ema21[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals