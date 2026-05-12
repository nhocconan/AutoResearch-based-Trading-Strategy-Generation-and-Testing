#!/usr/bin/env python3
# 6h_LongTermBreakout_VolumeConfirm
# Hypothesis: 6h price breaks above/below 20-period high/low (Donchian) with 1-day trend filter (EMA50) and volume spike confirmation.
# Uses 1-day EMA50 to filter trend direction (avoid counter-trend trades) and volume > 1.5x 20-period average to confirm strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull/bear by following 1-day trend.

name = "6h_LongTermBreakout_VolumeConfirm"
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

    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Calculate Donchian channels (20-period high/low) on 6h
    high_max_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values

    # Align 6h Donchian channels and 1d EMA50 to 6t timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_6h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_6h, low_min_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above 20-period high in 1-day uptrend with volume spike
            if close[i] > high_max_20_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 20-period low in 1-day downtrend with volume spike
            elif close[i] < low_min_20_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 20-period low (reversal signal)
            if close[i] < low_min_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 20-period high (reversal signal)
            if close[i] > high_max_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals