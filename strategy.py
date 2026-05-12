#!/usr/bin/env python3
# 6h_Donchian20_WeeklyTrend_DailyVolume
# Hypothesis: 6h Donchian(20) breakout aligned with weekly trend and daily volume confirmation.
# Weekly trend filter ensures trades follow long-term direction, reducing false breakouts in chop.
# Daily volume confirmation adds conviction to breakouts. Designed for 12-37 trades/year per symbol.
# Works in bull and bear via weekly trend filter (avoids counter-trend breakouts).

name = "6h_Donchian20_WeeklyTrend_DailyVolume"
timeframe = "6h"
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
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    # Weekly EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)

    # Daily volume average for confirmation (5-day average)
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=5, min_periods=5).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)

    # Donchian(20) on 6h data
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(highest[i]) or np.isnan(lowest[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filter from weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]

        # Volume confirmation: current volume > 1.5x daily average
        volume_ok = volume[i] > (1.5 * vol_avg_1d_aligned[i])

        if position == 0:
            # LONG: Close breaks above Donchian high AND above weekly EMA20 AND volume
            if close[i] > highest[i] and price_above_weekly_ema and volume_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Donchian low AND below weekly EMA20 AND volume
            elif close[i] < lowest[i] and price_below_weekly_ema and volume_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls back below Donchian low OR weekly trend turns down
            if close[i] < lowest[i] or not price_above_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises back above Donchian high OR weekly trend turns up
            if close[i] > highest[i] or not price_below_weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals