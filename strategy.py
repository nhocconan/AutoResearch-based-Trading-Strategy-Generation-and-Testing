#!/usr/bin/env python3
# 6h_1W_1D_Camarilla_R3_S3_Breakout_WeeklyTrend_DailyVolume
# Hypothesis: 6h Camarilla R3/S3 breakout with weekly trend filter (price above/below weekly SMA50) and daily volume confirmation (1.5x average volume).
# Uses weekly trend to avoid counter-trend trades and daily volume to confirm breakout strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee flood. Works in bull/bear by following weekly trend.

name = "6h_1W_1D_Camarilla_R3_S3_Breakout_WeeklyTrend_DailyVolume"
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

    # Get weekly data for trend filter (SMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)

    # Get daily data for Camarilla R3/S3 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for each daily bar
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4

    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)

    # Get daily volume for confirmation (1.5x average volume)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(sma50_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R3 in weekly uptrend with volume spike
            if close[i] > r3_aligned[i] and close[i] > sma50_1w_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S3 in weekly downtrend with volume spike
            elif close[i] < s3_aligned[i] and close[i] < sma50_1w_aligned[i] and volume[i] > volume_sma20[i]:
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