#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_1wTrend_1dVolumeSpike
# Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA trend filter and daily volume spike confirmation.
# The weekly EMA provides trend direction to avoid counter-trend trades, while volume spikes confirm breakout strength.
# Designed for 30-100 total trades over 4 years (7-25/year) to minimize fee drag. Works in bull/bear by following weekly trend.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_1dVolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:  # Need enough data for weekly trend
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for price action and volume
    df_1d = prices  # Already at daily timeframe
    if len(df_1d) < 2:
        return np.zeros(n)

    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate Camarilla levels for previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Previous day's typical price
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = typical_price[0]  # First value
    # Previous day's range
    prev_range = high - low
    prev_range = np.roll(prev_range, 1)
    prev_range[0] = prev_range[0] if len(prev_range) > 1 else 0.0  # First value

    # Camarilla levels
    camarilla_r3 = prev_typical + 1.1 * (prev_range / 2)
    camarilla_s3 = prev_typical - 1.1 * (prev_range / 2)

    # Calculate weekly EMA for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate daily volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Camarilla R3 in weekly uptrend with volume spike
            if close[i] > camarilla_r3[i] and close[i] > ema20_1w_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Camarilla S3 in weekly downtrend with volume spike
            elif close[i] < camarilla_s3[i] and close[i] < ema20_1w_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S3 (reversal signal)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R3 (reversal signal)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals