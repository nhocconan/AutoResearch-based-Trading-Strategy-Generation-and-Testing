#!/usr/bin/env python3
# 4h_ThreeTier_Trend_Confirmation_v1
# Hypothesis: Three-tier confirmation system using 1d MACD trend, 4h Donchian breakout, and volume spike.
# Uses 1d MACD histogram for trend filter (avoids whipsaw in ranging markets),
# 4h Donchian(20) breakout for entry signal, and volume > 2x SMA20 for confirmation.
# Designed for low trade frequency (<30/year) to minimize fee decay while capturing
# strong trending moves in both bull and bear markets. Exit when price retests
# Donchian midpoint or MACD histogram reverses.

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

    # Get 1d data for MACD trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d MACD histogram (12,26,9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_line - signal_line
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)

    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2

    # Volume confirmation: 2x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        macd_hist_aligned_val = macd_hist_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(macd_hist_aligned_val) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian HIGH + volume spike + bullish MACD histogram
            if (close[i] > donchian_high[i] and
                volume[i] > volume_threshold[i] and
                macd_hist_aligned_val > 0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian LOW + volume spike + bearish MACD histogram
            elif (close[i] < donchian_low[i] and
                  volume[i] > volume_threshold[i] and
                  macd_hist_aligned_val < 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price retests Donchian MID OR MACD histogram turns bearish
            if close[i] < donchian_mid[i] or macd_hist_aligned_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price retests Donchian MID OR MACD histogram turns bullish
            if close[i] > donchian_mid[i] or macd_hist_aligned_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals