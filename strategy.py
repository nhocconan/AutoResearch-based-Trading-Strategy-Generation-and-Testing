#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_1wTrend_Filter
# Hypothesis: Uses weekly Donchian breakout on 1d timeframe with 1w trend filter (EMA50) and volume confirmation (1.5x SMA20).
# Donchian breakouts capture momentum; weekly EMA50 filters for trend alignment to avoid counter-trend trades.
# Volume confirmation reduces false breakouts. Designed for low trade frequency (7-25/year) to minimize fee drag.
# Works in bull markets (breakouts) and bear markets (short breakdowns) with trend filter.

name = "1d_WeeklyDonchian_Breakout_1wTrend_Filter"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current daily bar
        ema50_aligned = ema50_1w_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_aligned) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + volume spike (1.5x) + weekly uptrend
            if (close[i] > donchian_high[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume spike (1.5x) + weekly downtrend
            elif (close[i] < donchian_low[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian low OR weekly trend turns down
            if close[i] < donchian_low[i] or close[i] < ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian high OR weekly trend turns up
            if close[i] > donchian_high[i] or close[i] > ema50_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals