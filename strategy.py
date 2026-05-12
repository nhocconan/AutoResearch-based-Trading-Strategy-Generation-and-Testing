#!/usr/bin/env python3
# 1d_Donchian20_Trend1w_Volume
# Hypothesis: Daily Donchian(20) breakout with 1-week trend filter and volume spike confirmation.
# Designed to work in both bull and bear markets by trading only in the direction of the 1-week trend.
# Uses volume confirmation (2.0x 20-day average volume) to filter breakouts and reduce false signals.
# Exit when price crosses below/above the 20-day EMA, providing a clean trend-following exit.
# Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag and improve generalization.

name = "1d_Donchian20_Trend1w_Volume"
timeframe = "1d"
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

    # Get 1d data for Donchian and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 1-week EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate Donchian channels (20-period) on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values

    # Calculate 20-day EMA for exit condition
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate 20-day volume average for volume confirmation
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_ma20 * 2.0  # Require 2.0x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(ema20_1d[i]) or
            np.isnan(volume_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high in 1w uptrend with volume spike
            if close[i] > donchian_high[i] and close[i] > ema20_1w_aligned[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low in 1w downtrend with volume spike
            elif close[i] < donchian_low[i] and close[i] < ema20_1w_aligned[i] and volume[i] > volume_spike_threshold[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 20-day EMA
            if close[i] < ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 20-day EMA
            if close[i] > ema20_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals