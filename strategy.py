#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot levels from daily timeframe combined with 1d EMA trend and volume
# confirmation work in both bull and bear markets. Breakouts at R1/S1 levels capture momentum,
# while the 1d EMA filter ensures trading with higher timeframe trend. Volume filter reduces
# false breakouts. Designed for ~25-35 trades/year to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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

    # Get 1d data for Camarilla pivot calculation and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Pivot = (High + Low + Close) / 3
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12

    # Align Camarilla levels to 4h timeframe (available after daily bar closes)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 1.8x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.8

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after indicators need 20 bars
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + volume + 1d uptrend
            if (close[i] > r1_1d_aligned[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume + 1d downtrend
            elif (close[i] < s1_1d_aligned[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below EMA34 OR below S1 (re-entry level)
            if close[i] < ema34_1d_aligned[i] or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above EMA34 OR above R1 (re-entry level)
            if close[i] > ema34_1d_aligned[i] or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals