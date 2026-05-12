#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA trend filter and volume spike confirmation.
# The 1d EMA provides trend direction to avoid counter-trend trades, while volume spikes confirm breakout strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull/bear by following 1d trend.
# Uses Camarilla pivot levels calculated from 1d high/low/close and 12h price action for breakouts.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 12h data for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Get 1d data for Camarilla pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for R3 and S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2

    # Calculate 1d EMA for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Align Camarilla levels and EMA to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 12h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Camarilla R3 in 1d uptrend with volume spike
            if close[i] > camarilla_r3_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Camarilla S3 in 1d downtrend with volume spike
            elif close[i] < camarilla_s3_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S3 (reversal signal)
            if close[i] < camarilla_s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R3 (reversal signal)
            if close[i] > camarilla_r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals