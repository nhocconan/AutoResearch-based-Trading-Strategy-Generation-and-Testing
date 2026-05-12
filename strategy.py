#!/usr/bin/env python3
# 6h_1d_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation (2x average volume).
# R3/S3 levels represent stronger support/resistance than R1/S1, reducing false breakouts.
# Trend filter ensures trades align with daily trend, avoiding counter-trend whipsaws.
# Volume spike confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear by following 1d trend, with tight entries to minimize fee drag.

name = "6h_1d_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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

    # Get 1d data for Camarilla pivot levels (R3, S3) and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4

    # Calculate EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Align 1d indicators to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0  # Require 2x average volume for confirmation

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema34_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R3 in 1d uptrend with volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_aligned[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 in 1d downtrend with volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_aligned[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 (reversal signal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 (reversal signal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals