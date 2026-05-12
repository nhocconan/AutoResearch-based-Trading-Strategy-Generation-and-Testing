#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_Volume
# Hypothesis: Weekly Donchian channel breakouts on daily timeframe, confirmed by volume spike.
# Weekly highs/lows act as strong support/resistance levels. Breakouts with volume indicate
# institutional participation. Works in both bull and bear markets by capturing momentum
# shifts. Low frequency (target 10-25 trades/year) minimizes fee drag.

name = "1d_WeeklyDonchian_Breakout_Volume"
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

    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)

    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values

    # Calculate 20-period weekly Donchian channels
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    weekly_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    weekly_low_20 = low_series.rolling(window=20, min_periods=20).min().values

    # Align weekly Donchian levels to daily timeframe
    weekly_high_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high_20)
    weekly_low_20_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low_20)

    # Volume spike: 2x 20-day SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):  # Start after 20-week lookback + buffer
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_20_aligned[i]) or np.isnan(weekly_low_20_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Donchian high with volume spike
            if close[i] > weekly_high_20_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low with volume spike
            elif close[i] < weekly_low_20_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low
            if close[i] < weekly_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high
            if close[i] > weekly_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals