#!/usr/bin/env python3
# 12h_1W_1D_Camarilla_R1_S1_Breakout_Trend_Volume
# Hypothesis: Trade breakouts from weekly and daily Camarilla R1/S1 levels in the direction of the weekly trend, with volume confirmation on the 12h timeframe. Uses weekly trend filter to capture major trends and daily levels for precise entry, reducing false signals. Targets 50-150 trades over 4 years to avoid fee drag while maintaining edge in both bull and bear markets by following higher-timeframe trend.

name = "12h_1W_1D_Camarilla_R1_S1_Breakout_Trend_Volume"
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

    # Get weekly data for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Get daily data for additional confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate weekly Camarilla levels (R1, S1) from previous week
    camarilla_range_1w = high_1w - low_1w
    r1_1w = close_1w + 1.1 * camarilla_range_1w / 12
    s1_1w = close_1w - 1.1 * camarilla_range_1w / 12

    # Calculate daily Camarilla levels (R1, S1) from previous day
    camarilla_range_1d = high_1d - low_1d
    r1_1d = close_1d + 1.1 * camarilla_range_1d / 12
    s1_1d = close_1d - 1.1 * camarilla_range_1d / 12

    # Align weekly and daily levels to 12h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate 12h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above weekly R1 and daily R1 in weekly uptrend with volume spike
            if (close[i] > r1_1w_aligned[i] and close[i] > r1_1d_aligned[i] and
                close[i] > ema34_1w_aligned[i] and volume[i] > volume_spike_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S1 and daily S1 in weekly downtrend with volume spike
            elif (close[i] < s1_1w_aligned[i] and close[i] < s1_1d_aligned[i] and
                  close[i] < ema34_1w_aligned[i] and volume[i] > volume_spike_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly EMA34 (trend change) or below daily S1
            if close[i] < ema34_1w_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly EMA34 (trend change) or above daily R1
            if close[i] > ema34_1w_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals