#!/usr/bin/env python3
"""
12h_Weekly_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirmation
Hypothesis: Weekly Camarilla R3/S3 levels act as strong support/resistance.
Breakouts above R3 or below S3 with weekly trend alignment (price above/below 200 EMA)
and volume confirmation capture institutional moves. Designed for low-frequency,
high-probability setups to minimize fee drag in both bull and bear markets.
"""

name = "12h_Weekly_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly Camarilla levels (based on prior week's range)
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r3 = pivot + (range_1w * 1.1 / 2)
    s3 = pivot - (range_1w * 1.1 / 2)

    # Weekly trend filter: 200-period EMA on weekly close
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values

    # Align weekly levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Volume spike: volume > 2.0 * 24-period average (~12 days at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + price breaks above R3 + volume spike
            if close[i] > ema200_1w_aligned[i] and close[i] > r3_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price breaks below S3 + volume spike
            elif close[i] < ema200_1w_aligned[i] and close[i] < s3_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below R3 or trend turns bearish
            if close[i] < r3_aligned[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above S3 or trend turns bullish
            if close[i] > s3_aligned[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals