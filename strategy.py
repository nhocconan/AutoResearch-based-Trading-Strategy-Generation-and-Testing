#!/usr/bin/env python3
# 1D_Donchian20_Trend_1w_Signal
# Hypothesis: Daily Donchian(20) breakout filtered by 1-week trend direction.
# Long when price breaks above 20-day high and 1-week close > 1-week open (bullish week).
# Short when price breaks below 20-day low and 1-week close < 1-week open (bearish week).
# Uses weekly trend to avoid counter-trend trades, reducing false signals in ranging markets.
# Target: 15-25 trades/year per symbol to minimize fee drag and improve generalization.

name = "1D_Donchian20_Trend_1w_Signal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate Donchian channels (20-period) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Weekly trend: bullish if weekly close > weekly open, bearish if close < open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bearish = weekly_close < weekly_open

    # Align weekly trend to daily timeframe (only use completed weekly bars)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if Donchian values are not ready
        if np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above 20-day high in bullish weekly trend
            if close[i] > high_20[i] and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below 20-day low in bearish weekly trend
            elif close[i] < low_20[i] and weekly_bearish_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-day low
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-day high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals