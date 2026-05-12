#!/usr/bin/env python3
# 4h_WeeklyTrend_DailyBreakout
# Hypothesis: Trade daily breakouts above/below daily high/low of the previous day only when the weekly trend is aligned.
# Weekly trend filter (EMA34) reduces whipsaws in range markets and ensures trades follow the dominant trend.
# Breakouts capture momentum; weekly filter improves win rate in both bull and bear markets.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "4h_WeeklyTrend_DailyBreakout"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Daily high/low of previous day (for breakout levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    # Fill first value with same day's high/low to avoid NaN
    prev_day_high[0] = df_1d['high'].values[0]
    prev_day_low[0] = df_1d['low'].values[0]

    # Align daily levels to 4h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(prev_day_high_aligned[i]) or
            np.isnan(prev_day_low_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]

        if position == 0:
            # LONG: Price breaks above previous day's high with weekly uptrend and volume
            if close[i] > prev_day_high_aligned[i] and bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below previous day's low with weekly downtrend and volume
            elif close[i] < prev_day_low_aligned[i] and bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below previous day's low or weekly trend turns bearish
            if close[i] < prev_day_low_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above previous day's high or weekly trend turns bullish
            if close[i] > prev_day_high_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals