#!/usr/bin/env python3
# 4h_TRIX_Zero_Cross_With_Volume_Filter
# Hypothesis: TRIX (triple EMA) zero-line cross indicates momentum shift. 
# Long when TRIX crosses above zero with volume confirmation; short when crosses below zero.
# Volume filter ensures institutional participation. Works in bull/bear by following momentum.
# Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades.

name = "4h_TRIX_Zero_Cross_With_Volume_Filter"
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

    # TRIX calculation (15-period triple EMA)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value undefined

    # 12h EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume filter: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or 
            np.isnan(trix[i-1]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + above 12h EMA50 + volume filter
            if trix[i-1] <= 0 and trix[i] > 0 and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + below 12h EMA50 + volume filter
            elif trix[i-1] >= 0 and trix[i] < 0 and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or price below 12h EMA50
            if trix[i] < 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or price above 12h EMA50
            if trix[i] > 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals