#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: 6h Camarilla pivot breakout at R3/S3 levels with 1d EMA50 trend filter and volume spike confirmation.
# The Camarilla pivot system provides mathematically derived support/resistance levels.
# Breakouts at R3/S3 (stronger levels than R1/S1) with trend alignment and volume confirmation
# should capture strong momentum moves while avoiding false breakouts. Works in both bull and bear
# by following 1d trend direction. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
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

    # Get 6h data for Camarilla pivot calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values

    # Calculate Camarilla pivot levels for each 6h bar using previous bar's data
    # Camarilla formulas:
    # Pivot = (H + L + C) / 3
    # R3 = Pivot + (H - L) * 1.1000
    # S3 = Pivot - (H - L) * 1.1000
    # R4 = Pivot + (H - L) * 1.5000
    # S4 = Pivot - (H - L) * 1.5000
    
    pivot_6h = (high_6h + low_6h + close_6h) / 3.0
    range_6h = high_6h - low_6h
    r3_6h = pivot_6h + range_6h * 1.1000
    s3_6h = pivot_6h - range_6h * 1.1000
    r4_6h = pivot_6h + range_6h * 1.5000
    s4_6h = pivot_6h - range_6h * 1.5000

    # Align Camarilla levels to 6h timeframe (using previous bar's values)
    pivot_aligned = align_htf_to_ltf(prices, df_6h, pivot_6h)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4_6h)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4_6h)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R3 in 1d uptrend with volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 in 1d downtrend with volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
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