#!/usr/bin/env python3
# 6h_1D_SlopeOfLinearRegression_TrendFilter
# Hypothesis: Use the slope of linear regression over 20 periods on 1d closes as a trend filter for 6b breakouts.
# In strong trends (positive slope), buy breakouts above 20-period high; in weak trends (negative slope), sell breakdowns below 20-period low.
# The slope acts as a smoothed trend strength indicator, reducing whipsaws in sideways markets.
# Designed for 6h to limit trade frequency to 50-150 total trades over 4 years.

name = "6h_1D_SlopeOfLinearRegression_TrendFilter"
timeframe = "6h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate slope of linear regression over 20 periods on 1d closes
    def linreg_slope(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            slope = np.polyfit(x, y, 1)[0]
            result[i] = slope
        return result

    slope_20 = linreg_slope(df_1d['close'].values, 20)
    slope_20_aligned = align_htf_to_ltf(prices, df_1d, slope_20)

    # 20-period high and low on 6h for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(slope_20_aligned[i]) or np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter: positive slope = uptrend, negative slope = downtrend
        uptrend = slope_20_aligned[i] > 0
        downtrend = slope_20_aligned[i] < 0

        if position == 0:
            # LONG: Break above 20-period high in uptrend with volume confirmation
            if close[i] > high_20[i] and uptrend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 20-period low in downtrend with volume confirmation
            elif close[i] < low_20[i] and downtrend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below 20-period low or trend turns negative
            if close[i] < low_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above 20-period high or trend turns positive
            if close[i] > high_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals