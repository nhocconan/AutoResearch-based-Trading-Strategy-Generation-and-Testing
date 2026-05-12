#!/usr/bin/env python3
# 6h_Donchian_20_WeeklyTrend_DailyVolume
# Hypothesis: Use weekly trend (via weekly close > weekly SMA20) and daily volume breakout
# to enter Donchian(20) breakouts on 6h timeframe. Long when weekly uptrend, daily volume > 1.5x avg,
# and price breaks above 20-period high. Short when weekly downtrend, daily volume > 1.5x avg,
# and price breaks below 20-period low. Exit when price crosses the 20-period midpoint.
# Weekly trend filters noise, daily volume confirms momentum, Donchian provides clear breakout levels.
# Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets.
# Targets ~20-30 trades/year by requiring weekly trend alignment + volume surge + breakout.

name = "6h_Donchian_20_WeeklyTrend_DailyVolume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Weekly trend: close > SMA20
    weekly_sma20 = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    weekly_trend_up = df_1w['close'].values > weekly_sma20
    weekly_trend_down = df_1w['close'].values < weekly_sma20
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)

    # Daily volume: current volume > 1.5x average of last 5 days
    daily_volume_avg = pd.Series(df_1d['volume']).rolling(window=5, min_periods=5).mean().values
    daily_volume_ok = df_1d['volume'].values > (1.5 * daily_volume_avg)
    daily_volume_ok_aligned = align_htf_to_ltf(prices, df_1d, daily_volume_ok)

    # Donchian channel (20-period) on 6h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i]) or
            np.isnan(daily_volume_ok_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: weekly uptrend, volume breakout, price breaks above Donchian high
            if weekly_trend_up_aligned[i] and daily_volume_ok_aligned[i] and close[i] > high_20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: weekly downtrend, volume breakout, price breaks below Donchian low
            elif weekly_trend_down_aligned[i] and daily_volume_ok_aligned[i] and close[i] < low_20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below midpoint
            if close[i] < mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above midpoint
            if close[i] > mid_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals