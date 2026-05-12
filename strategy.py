#!/usr/bin/env python3
"""
12h_TRIX_ZeroCross_With_Volume_And_Trend_Filter
Hypothesis: TRIX (triple smoothed EMA) crossing above/below zero line indicates momentum shifts.
Combine with volume > 1.5x average and 1d EMA50 trend filter to capture strong momentum moves
while avoiding whipsaws. Works in both bull and bear markets by following momentum direction.
Target: 20-40 trades per year to minimize fee drag.
"""

name = "12h_TRIX_ZeroCross_With_Volume_And_Trend_Filter"
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
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate TRIX on 12h close: triple EMA then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100  # percentage change
    trix = trix.fillna(0).values

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(trix[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + uptrend + volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + downtrend + volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR trend turns down
            if (trix[i] < 0 and trix[i-1] >= 0) or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR trend turns up
            if (trix[i] > 0 and trix[i-1] <= 0) or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals