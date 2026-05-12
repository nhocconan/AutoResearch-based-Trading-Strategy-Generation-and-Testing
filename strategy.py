#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_Volume_1dTrend
Hypothesis: TRIX (12) zero-cross with volume confirmation and 1d EMA trend filter captures momentum shifts with low whipsaw. Works in both bull and bear markets by filtering counter-trend moves.
"""

name = "4h_TRIX_ZeroCross_Volume_1dTrend"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate TRIX (12) on close: EMA3 of EMA3 of EMA3, then percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # percent change

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 4h volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):
        trix_val = trix[i]
        ema34_val = ema34_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(trix_val) or np.isnan(ema34_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + uptrend + volume confirmation
            if trix_val > 0 and trix[i-1] <= 0 and close[i] > ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + downtrend + volume confirmation
            elif trix_val < 0 and trix[i-1] >= 0 and close[i] < ema34_val and volume[i] > vol_avg_val * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend breaks
            if trix_val < 0 or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend breaks
            if trix_val > 0 or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals