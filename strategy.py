#!/usr/bin/env python3
# 6h_LinearRegression_Trend_Follower
# Hypothesis: Use 6h linear regression slope to detect medium-term trend, confirmed by 12h trend (EMA100) and volume spikes (>1.5x 30-period average). Enter long when regression slope > 0 and price > 12h EMA100 with volume spike; short when regression slope < 0 and price < 12h EMA100 with volume spike. Exit on regression slope cross. Targets 10-30 trades/year to minimize fee drift and work in both bull/bear markets via trend filter.

name = "6h_LinearRegression_Trend_Follower"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    close_12h = df_12h['close'].values

    # Calculate linear regression slope on 6h (20-period)
    def linreg_slope(arr, window):
        slopes = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            if np.any(np.isnan(y)):
                continue
            slope = np.polyfit(x, y, 1)[0]
            slopes[i] = slope
        return slopes

    lr_slope = linreg_slope(close, 20)

    # 12h EMA100 for trend filter
    ema100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema100_12h)

    # Volume confirmation: volume > 1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(lr_slope[i]) or np.isnan(ema100_12h_aligned[i]) or 
            np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: LR slope > 0 (bullish trend) + price > 12h EMA100 + volume spike
            if (lr_slope[i] > 0 and 
                close[i] > ema100_12h_aligned[i] and
                volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: LR slope < 0 (bearish trend) + price < 12h EMA100 + volume spike
            elif (lr_slope[i] < 0 and 
                  close[i] < ema100_12h_aligned[i] and
                  volume[i] > vol_avg_30[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: LR slope turns negative
            if lr_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: LR slope turns positive
            if lr_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals