#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend_1d
# Hypothesis: 4-hour Donchian channel breakout with daily trend filter and volume confirmation.
# Works in bull markets by capturing upward breakouts and in bear markets by capturing downward breakdowns.
# Daily trend filter avoids whipsaws in range markets; volume confirms institutional interest.
# Designed for 20-50 trades per year to minimize fee drag.

name = "4h_Donchian_Breakout_Volume_Trend_1d"
timeframe = "4h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate 4h Donchian channels (20-period)
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Daily trend filter
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]

        if position == 0:
            # LONG: Price breaks above Donchian upper with bullish daily trend and volume confirmation
            if high[i] > high_4h[i-1] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower with bearish daily trend and volume confirmation
            elif low[i] < low_4h[i-1] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower or daily trend turns bearish
            if low[i] < low_4h[i-1] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper or daily trend turns bullish
            if high[i] > high_4h[i-1] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals