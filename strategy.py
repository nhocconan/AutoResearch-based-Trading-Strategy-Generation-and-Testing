#!/usr/bin/env python3
# 6h_1W_1D_Camarilla_R3_S3_Breakout_Trend_Filtered_v1
# Hypothesis: Weekly and daily trend alignment with Camarilla R3/S3 breakouts.
# Uses weekly trend (EMA34) for primary direction, daily trend (EMA34) for secondary confirmation,
# and breaks above daily R3 or below S3 with volume > 1.5x average for entry.
# Exits when price crosses the opposite level (R3 for longs, S3 for shorts).
# Designed for 6h timeframe to capture multi-day moves with low trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via trend-following breaks and in bear via short breakdowns.

name = "6h_1W_1D_Camarilla_R3_S3_Breakout_Trend_Filtered_v1"
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

    # Get weekly data for primary trend and context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Get daily data for Camarilla levels and secondary trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate weekly EMA34 for primary trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate daily EMA34 for secondary trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate daily Camarilla levels (R3, S3) from previous day
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range / 4
    s3 = close_1d - 1.1 * camarilla_range / 4

    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R3 with weekly and daily uptrend + volume spike
            if (close[i] > r3_aligned[i] and
                close[i] > ema34_1w_aligned[i] and
                close[i] > ema34_1d_aligned[i] and
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 with weekly and daily downtrend + volume spike
            elif (close[i] < s3_aligned[i] and
                  close[i] < ema34_1w_aligned[i] and
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 (mean reversion or failed breakout)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 (mean reversion or failed breakdown)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals