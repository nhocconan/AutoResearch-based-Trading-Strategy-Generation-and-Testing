#!/usr/bin/env python3
# 4h_ThreeTier_Trend_Confirmation_v1
# Hypothesis: Three-tier confirmation system for 4H timeframe:
# 1. Primary: 20-period Donchian breakout (price channel breakout)
# 2. Secondary: 12H EMA50 trend filter (higher timeframe trend alignment)
# 3. Tertiary: Volume surge (>1.5x 20-period SMA) for confirmation
# This creates a high-probability setup that works in both bull and bear markets
# by combining breakout momentum with trend alignment and volume validation.
# Target: 25-35 trades/year to minimize fee drag while capturing strong moves.

name = "4h_ThreeTier_Trend_Confirmation_v1"
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

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    close_12h = df_12h['close'].values

    # Calculate Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Calculate 12H EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after indicators need 20 bars
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian Upper + volume + 12H uptrend
            if (close[i] > highest_high[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian Lower + volume + 12H downtrend
            elif (close[i] < lowest_low[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian Middle OR 12H trend turns down
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_middle or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian Middle OR 12H trend turns up
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_middle or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals