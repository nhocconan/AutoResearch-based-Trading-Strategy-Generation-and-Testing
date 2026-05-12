#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation (1.5x average volume).
# Uses Camarilla pivot levels from 1d for structure, 1d EMA50 for trend direction to avoid counter-trend trades, and volume spikes to confirm breakout strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Works in bull/bear by following 1d trend.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
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

    # Get 12h data for Camarilla calculation (use high/low/close)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate Camarilla levels for each 12h bar
    # R1 = close + (high - low) * 1.12 / 12
    # S1 = close - (high - low) * 1.12 / 12
    r1_12h = close_12h + (high_12h - low_12h) * 1.12 / 12
    s1_12h = close_12h - (high_12h - low_12h) * 1.12 / 12

    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 12h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above R1 in 1d uptrend with volume spike
            if close[i] > r1_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below S1 in 1d downtrend with volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 (reversal signal)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 (reversal signal)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals