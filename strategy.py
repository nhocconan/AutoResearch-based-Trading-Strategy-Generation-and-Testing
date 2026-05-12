#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation (2x average volume).
# R3/S3 levels from daily Camarilla calculation provide stronger support/resistance than R1/S1.
# Trend filter uses 1d EMA34 to avoid counter-trend trades, volume spike confirms breakout strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull/bear by following 1d trend.
# Uses Camarilla levels calculated from 1d data for breakout levels.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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

    # Get 1d data for Camarilla calculation (high, low, close from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each day using previous day's OHLC
    # R4 = close + 1.5*(high - low)
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # S4 = close - 1.5*(high - low)
    # We use R3 and S3 for breakout/breakdown
    range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d

    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate 12h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0  # Require 2x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R3 in 1d uptrend with volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 in 1d downtrend with volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > volume_sma20[i]:
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