#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_VolumeFilter
# Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) combined with 1d trend filter.
# In bull markets (1d close > EMA50), go long when Bull Power turns positive; in bear markets (1d close < EMA50),
# go short when Bear Power turns negative. Volume confirmation ensures momentum legitimacy.
# Designed for 50-150 total trades over 4 years (12-37/year) on 6h timeframe. Works in both bull/bear by following 1d trend.
# Uses Elder Ray to detect institutional buying/selling pressure at micro-structure level.

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeFilter"
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

    # Get 6h data for EMA13 (Elder Ray base)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 13:
        return np.zeros(n)

    close_6h = df_6h['close'].values
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_6h_aligned = align_htf_to_ltf(prices, df_6h, ema13_6h)

    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_6h_aligned
    bear_power = low - ema13_6h_aligned

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (buying pressure) in 1d uptrend with volume spike
            if bull_power[i] > 0 and close[i] > ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (selling pressure) in 1d downtrend with volume spike
            elif bear_power[i] < 0 and close[i] < ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power turns negative (selling pressure emerges)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power turns positive (buying pressure emerges)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals