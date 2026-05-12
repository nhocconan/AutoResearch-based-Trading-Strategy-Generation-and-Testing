#!/usr/bin/env python3
# 1d_WeeklyDonchian_Breakout_1wEMA34_Trend_Volume
# Hypothesis: On daily timeframe, breakouts above/below weekly Donchian channels with volume confirmation
# and weekly EMA trend filter capture major trends in both bull and bear markets. Weekly EMA filter
# ensures alignment with higher timeframe trend, reducing whipsaws. Volume confirmation ensures
# breakouts are supported by participation. Designed for 1d timeframe to target 10-25 trades per year.

name = "1d_WeeklyDonchian_Breakout_1wEMA34_Trend_Volume"
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

    # Get weekly data for Donchian channels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values

    # Calculate weekly Donchian channels (20-period)
    # Upper band = max(high over past 20 weeks)
    # Lower band = min(low over past 20 weeks)
    high_series = pd.Series(high_1w)
    low_series = pd.Series(low_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values

    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Align weekly indicators to daily timeframe (using completed weekly bars)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: 1.8x 20-week SMA (higher threshold to reduce trades)
    volume_series_1w = pd.Series(volume_1w)
    volume_sma20_1w = volume_series_1w.rolling(window=20, min_periods=20).mean().values
    volume_threshold_1w = volume_sma20_1w * 1.8
    volume_threshold_aligned = align_htf_to_ltf(prices, df_1w, volume_threshold_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after Donchian needs 20 weeks
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_threshold_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above weekly Donchian high with volume confirmation and uptrend
            if (close[i] > donchian_high_aligned[i] and
                volume[i] > volume_threshold_aligned[i] and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly Donchian low with volume confirmation and downtrend
            elif (close[i] < donchian_low_aligned[i] and
                  volume[i] > volume_threshold_aligned[i] and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below weekly Donchian low (opposite side)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above weekly Donchian high (opposite side)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals