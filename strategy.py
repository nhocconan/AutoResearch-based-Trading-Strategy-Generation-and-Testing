#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakouts from 1d pivot levels, filtered by 1d EMA34 trend and volume spikes, capture momentum on 12h timeframe. Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend) markets. Designed for low trade frequency (~15-25/year) to minimize fee drag.
"""

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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

    # Get 1d data for trend filter and Camarilla levels (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Camarilla pivot levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_S3 = close_1d - (high_1d - low_1d) * 1.1 / 4

    # Align Camarilla levels and trend to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    trend_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or np.isnan(trend_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above Camarilla R3 + 1d uptrend + volume spike
            if close[i] > camarilla_R3_aligned[i] and close[i] > trend_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below Camarilla S3 + 1d downtrend + volume spike
            elif close[i] < camarilla_S3_aligned[i] and close[i] < trend_aligned[i] and volume[i] > vol_avg_20[i] * 2:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below Camarilla S3 or 1d trend turns down
            if close[i] < camarilla_S3_aligned[i] or close[i] < trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above Camarilla R3 or 1d trend turns up
            if close[i] > camarilla_R3_aligned[i] or close[i] > trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals