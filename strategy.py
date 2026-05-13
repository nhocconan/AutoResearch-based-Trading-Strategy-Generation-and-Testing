#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1_S1_1dTrend_VolumeConfirm
Hypothesis: Camarilla pivot R1/S1 levels on 12h timeframe with daily trend filter and volume confirmation
capture significant trend moves while avoiding whipsaw. Daily trend ensures alignment with higher-timeframe
momentum, reducing false signals. Volume confirms breakout strength. Uses 12h timeframe to target 15-30 trades/year.
"""

name = "12h_Camarilla_Pivot_R1_S1_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate daily EMA20 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate Camarilla pivot levels for 12h timeframe using previous bar's OHLC
    pivot = (high + low + close) / 3.0
    r1 = close + (high - low) * 1.1 / 12
    s1 = close - (high - low) * 1.1 / 12

    # Shift to get previous bar's levels (no look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(r1_prev[i]) or np.isnan(s1_prev[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with volume spike and daily uptrend
            if close[i] > r1_prev[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and daily downtrend
            elif close[i] < s1_prev[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below S1 or daily trend turns down
            if close[i] < s1_prev[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above R1 or daily trend turns up
            if close[i] > r1_prev[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals