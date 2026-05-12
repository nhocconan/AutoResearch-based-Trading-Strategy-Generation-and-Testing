#!/usr/bin/env python3
# 6h_Donchian_20_WeeklyPivotDirection_Volume
# Hypothesis: 6s Donchian(20) breakout in direction of weekly pivot (weekly high/low), with volume confirmation.
# Uses weekly pivot (weekly high/low) as trend filter - long only when price > weekly midpoint, short only when price < weekly midpoint.
# Weekly pivot provides longer-term bias, reducing false breakouts in sideways markets.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to stay under fee drag threshold.

name = "6h_Donchian_20_WeeklyPivotDirection_Volume"
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

    # Get daily data for calculating weekly pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of data
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate weekly pivot points: weekly high and weekly low
    # We'll use 5-day rolling window to approximate weekly (5 trading days)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_midpoint = (weekly_high + weekly_low) / 2.0

    # Align weekly data to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1d, weekly_low)
    weekly_midpoint_aligned = align_htf_to_ltf(prices, df_1d, weekly_midpoint)

    # Calculate Donchian(20) channels on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: volume > 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or
            np.isnan(weekly_midpoint_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian high AND price > weekly midpoint (uptrend bias) AND volume confirmation
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_midpoint_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low AND price < weekly midpoint (downtrend bias) AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_midpoint_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches or crosses below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches or crosses above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals