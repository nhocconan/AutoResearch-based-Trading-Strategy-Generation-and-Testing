#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Daily chart breakouts above R1 or below S1 with weekly trend filter and volume confirmation capture institutional moves in both bull and bear markets. Weekly trend ensures alignment with higher timeframe momentum while reducing whipsaws. Designed for low turnover (target: 15-25 trades/year) to minimize fee drag.
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate daily Camarilla levels: R1, S1
    high_1d = high
    low_1d = low
    close_1d = close
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12

    # Align weekly close to daily (wait for weekly bar to close)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)

    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: volume > 1.8x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + weekly uptrend + volume spike
            if close[i] > camarilla_r1[i] and close[i] > ema34_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.8:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + weekly downtrend + volume spike
            elif close[i] < camarilla_s1[i] and close[i] < ema34_1w_aligned[i] and volume[i] > vol_avg_20[i] * 1.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below S1 or weekly trend turns down
            if close[i] < camarilla_s1[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above R1 or weekly trend turns up
            if close[i] > camarilla_r1[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals