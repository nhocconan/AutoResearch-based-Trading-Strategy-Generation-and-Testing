#!/usr/bin/env python3
"""
6h_LinearRegressionBreakout_1dTrend
Hypothesis: Linear regression slope on 6h closes identifies emerging trends, filtered by 1d EMA50 trend direction. Long when 6h LR slope > 0 and 1d uptrend; short when slope < 0 and 1d downtrend. Uses volume confirmation (volume > 1.5x 20-period average) to avoid false breakouts. Works in bull/bear by following the 1d trend.
"""

name = "6h_LinearRegressionBreakout_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from scipy import stats
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Linear regression slope over 20 periods (6h timeframe)
    def linreg_slope(arr, window):
        slope = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            if np.all(np.isnan(y)):
                continue
            slope[i] = stats.linregress(x, y)[0]
        return slope

    lr_slope = linreg_slope(close, 20)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(lr_slope[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Positive LR slope + 1d uptrend + volume spike
            if lr_slope[i] > 0 and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Negative LR slope + 1d downtrend + volume spike
            elif lr_slope[i] < 0 and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: LR slope turns negative or 1d trend turns down
            if lr_slope[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: LR slope turns positive or 1d trend turns up
            if lr_slope[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals