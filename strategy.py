#!/usr/bin/env python3
# 6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation
# Hypothesis: Donchian channel (20) breakouts on 6h timeframe, filtered by weekly pivot direction (from weekly close),
# and confirmed by volume spikes, provide high-probability entries in both bull and bear markets.
# Weekly pivot direction acts as a regime filter: bullish when weekly close > weekly pivot, bearish when below.
# This avoids counter-trend trades and reduces whipsaw. Volume spike ensures institutional participation.
# Targets 15-35 trades per year per symbol to minimize fee drag.

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot direction (weekly close vs weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Weekly pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w

    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))

    # Donchian channel (20) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values

    # Volume spike: volume > 2.0 * 20-period average (~6.6 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly bullish + breakout above Donchian high + volume spike
            if weekly_bullish_aligned[i] > 0.5 and close[i] > highest_high[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly bearish + breakdown below Donchian low + volume spike
            elif weekly_bearish_aligned[i] > 0.5 and close[i] < lowest_low[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or weekly bias turns bearish
            if close[i] < lowest_low[i] or weekly_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or weekly bias turns bullish
            if close[i] > highest_high[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals