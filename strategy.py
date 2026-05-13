#!/usr/bin/env python3
# 1D_WKLY_CAMARILLA_PIVOT_BREAKOUT
# Hypothesis: Buy when price breaks above weekly Camarilla R3 with daily trend filter and volume spike; sell when price breaks below weekly S3 with daily downtrend and volume spike. Exit when price crosses weekly pivot point (mean reversion). Uses weekly levels to avoid overtrading and focuses on strong breakouts in trending markets. Target: 10-30 trades/year on 1d to minimize fee drag while capturing strong moves in both bull and bear markets.

name = "1D_WKLY_CAMARILLA_PIVOT_BREAKOUT"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly Camarilla pivot levels
    pivot = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r3 = pivot + (range_1w * 1.1 / 2)
    s3 = pivot - (range_1w * 1.1 / 2)

    # Align weekly levels to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)

    # Daily EMA34 for trend filter
    ema34_d = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Volume confirmation: volume > 2.0x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema34_d[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above weekly R3 + price > daily EMA34 + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_d[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below weekly S3 + price < daily EMA34 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_d[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below weekly pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above weekly pivot
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals