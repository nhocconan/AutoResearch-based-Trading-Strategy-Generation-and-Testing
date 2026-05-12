#!/usr/bin/env python3
"""
6h_ElderRay_1dTrend_Filtered
Hypothesis: Elder Ray Index (bull/bear power) confirms institutional buying/selling pressure.
In bull markets: buy when bull power > 0 and rising, EMA13 > EMA34, price above EMA13.
In bear markets: sell when bear power < 0 and falling, EMA13 < EMA34, price below EMA13.
Uses 1d EMA34 as trend filter to avoid counter-trend trades. Targets 15-25 trades/year.
Works in both bull (captures rallies) and bear (captures rallies within downtrend).
"""

name = "6h_ElderRay_1dTrend_Filtered"
timeframe = "6h"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate 13-period EMA for Elder Ray (using Wilder's smoothing)
    def ema_wilder(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(arr[:period])
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result

    ema13 = ema_wilder(close, 13)
    ema34 = ema_wilder(close, 34)

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13

    # 1d EMA34 for trend filter
    ema34_1d = ema_wilder(close_1d, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required value is NaN
        if (np.isnan(ema13[i]) or np.isnan(ema34[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull power positive AND rising, price > EMA13, 1d uptrend
            if (bull_power[i] > 0 and 
                i > 34 and bull_power[i] > bull_power[i-1] and
                close[i] > ema13[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear power negative AND falling, price < EMA13, 1d downtrend
            elif (bear_power[i] < 0 and 
                  i > 34 and bear_power[i] < bear_power[i-1] and
                  close[i] < ema13[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull power turns negative OR price < EMA13 OR 1d trend turns down
            if (bull_power[i] <= 0 or 
                close[i] < ema13[i] or
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear power turns positive OR price > EMA13 OR 1d trend turns up
            if (bear_power[i] >= 0 or 
                close[i] > ema13[i] or
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals