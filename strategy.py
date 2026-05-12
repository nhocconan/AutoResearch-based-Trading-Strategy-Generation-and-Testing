#!/usr/bin/env python3
"""
4h_Donchian_Breakout_12hTrend_Volume
Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
Designed to work in both bull and bear markets via strict entry conditions and trend alignment.
"""

name = "4h_Donchian_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)

    # Volume confirmation: 1.8x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.8

    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above 20-period high + volume spike + 12h uptrend
            if (high[i] > high_max[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + volume spike + 12h downtrend
            elif (low[i] < low_min[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low
            if low[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high
            if high[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals