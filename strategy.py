#!/usr/bin/env python3
# 6h_LinearRegression_Trend_With_Volume_Filter
# Hypothesis: Linear regression slope of closing prices over 30 periods determines trend direction on 6h.
# Enter long when slope > 0 and volume > 1.5x 20-period average; short when slope < 0 with volume filter.
# Exit when slope changes sign. Uses price action trend with volume confirmation to avoid whipsaws.
# Designed for 6h timeframe to target 12-37 trades/year, working in both bull and bear markets via trend filter.

name = "6h_LinearRegression_Trend_With_Volume_Filter"
timeframe = "6h"
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

    # Linear regression slope of close over 30 periods
    def linreg_slope(arr, window):
        slopes = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                continue
            # Use valid points only
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                continue
            x_valid = x[mask]
            y_valid = y[mask]
            slope = np.polyfit(x_valid, y_valid, 1)[0]
            slopes[i] = slope
        return slopes

    slope_30 = linreg_slope(close, 30)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if np.isnan(slope_30[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: upward slope + volume filter
            if slope_30[i] > 0 and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: downward slope + volume filter
            elif slope_30[i] < 0 and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: slope turns negative
            if slope_30[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: slope turns positive
            if slope_30[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals