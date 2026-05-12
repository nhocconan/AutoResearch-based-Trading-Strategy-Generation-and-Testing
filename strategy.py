#!/usr/bin/env python3
# 1d_Donchian_20_Breakout_1wTrend_Volume
# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume spike confirmation.
# Long when price breaks above 20-day high with weekly uptrend (price > weekly EMA50) and volume spike.
# Short when price breaks below 20-day low with weekly downtrend (price < weekly EMA50) and volume spike.
# Exit on opposite Donchian level touch. Designed for low trade frequency (7-25/year) to avoid fee drag.
# Works in bull/bear markets by following weekly EMA trend direction.

name = "1d_Donchian_20_Breakout_1wTrend_Volume"
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

    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Donchian(20) channels: based on past 20 days (excluding current)
    # Highest high of past 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    # Lowest low of past 20 days
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values

    # Align Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate volume spike threshold (2.0x 20-day SMA on daily)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian high with weekly uptrend and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low with weekly downtrend and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches or crosses below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches or crosses above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals