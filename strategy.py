#!/usr/bin/env python3
# 1d_WeeklyTrend_DailyPullback
# Hypothesis: In the direction of the weekly trend, buy pullbacks to daily support/resistance
# with volume confirmation. Weekly trend filters out counter-trend noise, while daily
# pullbacks provide better entry prices. Works in bull (buy dips in uptrend) and bear
# (sell rallies in downtrend) by trading with the weekly trend. Target: 10-20 trades/year.

name = "1d_WeeklyTrend_DailyPullback"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Get daily data for pullback and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Weekly trend: 34-period EMA on weekly close
    weekly_ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_uptrend = close_1w > weekly_ema34
    weekly_downtrend = close_1w < weekly_ema34

    # Daily pullback: price near 20-period EMA on daily timeframe
    daily_ema20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Allow 1% deviation from EMA for pullback
    pullback_threshold = 0.01
    near_ema = np.abs(close_1d - daily_ema20) / daily_ema20 < pullback_threshold

    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)

    # Align weekly and daily indicators to 1d timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    near_ema_aligned = align_htf_to_ltf(prices, df_1d, near_ema)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after weekly EMA warmup
        # Get aligned values
        uptrend = weekly_uptrend_aligned[i]
        downtrend = weekly_downtrend_aligned[i]
        pullback = near_ema_aligned[i]
        vol_spike = volume_spike_aligned[i]

        if position == 0:
            # LONG: Weekly uptrend + daily pullback to EMA + volume spike
            if uptrend and pullback and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + daily pullback to EMA + volume spike
            elif downtrend and pullback and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price moves 1.5% above EMA or weekly trend changes
            if (close[i] > daily_ema20[i] * 1.015 or not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price moves 1.5% below EMA or weekly trend changes
            if (close[i] < daily_ema20[i] * 0.985 or not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals