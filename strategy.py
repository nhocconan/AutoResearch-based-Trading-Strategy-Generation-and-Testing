#!/usr/bin/env python3
# 4h_SuperTrend_12hTrend_Volume
# Hypothesis: 4h SuperTrend trend following with 12h EMA trend filter and volume confirmation.
# The 12h EMA provides higher timeframe trend direction to avoid counter-trend trades,
# while volume spikes confirm momentum. SuperTrend adapts to volatility via ATR.
# Designed for 75-200 total trades over 4 years (19-50/year) to minimize fee drift.
# Works in bull/bear by following 12h trend.

name = "4h_SuperTrend_12hTrend_Volume"
timeframe = "4h"
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

    # Get 4h data for SuperTrend calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate ATR for SuperTrend
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Calculate SuperTrend
    upper_band = (high_4h + low_4h) / 2 + 3 * atr
    lower_band = (high_4h + low_4h) / 2 - 3 * atr

    upper_band_final = np.zeros_like(upper_band)
    lower_band_final = np.zeros_like(lower_band)
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend

    # Initialize first values
    upper_band_final[0] = upper_band[0]
    lower_band_final[0] = lower_band[0]
    supertrend[0] = upper_band[0]
    direction[0] = 1

    for i in range(1, len(close_4h)):
        # Update upper and lower bands
        if upper_band[i] <= upper_band_final[i-1] or close_4h[i-1] > upper_band_final[i-1]:
            upper_band_final[i] = upper_band[i]
        else:
            upper_band_final[i] = upper_band_final[i-1]

        if lower_band[i] >= lower_band_final[i-1] or close_4h[i-1] < lower_band_final[i-1]:
            lower_band_final[i] = lower_band[i]
        else:
            lower_band_final[i] = lower_band_final[i-1]

        # Determine trend direction
        if close_4h[i] > upper_band_final[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band_final[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        # Set SuperTrend value
        if direction[i] == 1:
            supertrend[i] = lower_band_final[i]
        else:
            supertrend[i] = upper_band_final[i]

    # Align SuperTrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: SuperTrend uptrend, 12h EMA uptrend, volume spike
            if direction_aligned[i] == 1 and ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: SuperTrend downtrend, 12h EMA downtrend, volume spike
            elif direction_aligned[i] == -1 and ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: SuperTrend turns down
            if direction_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: SuperTrend turns up
            if direction_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals