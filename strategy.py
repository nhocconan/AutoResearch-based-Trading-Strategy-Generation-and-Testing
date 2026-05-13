#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Camarilla pivot reversals at S1/R1 levels with 12h EMA50 trend filter and volume confirmation
# capture mean-reversion bounces in ranging markets while avoiding false signals. 12h EMA ensures alignment
# with higher-timeframe momentum, reducing counter-trend trades. Volume confirms rejection strength.
# Target: 25-50 trades/year (100-200 total over 4 years) to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
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

    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate Camarilla levels (S1, R1) on 12h data
    camarilla_S1 = np.full(n, np.nan)
    camarilla_R1 = np.full(n, np.nan)
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    camarilla_S1 = pivot - (1.1 * range_12h / 12)
    camarilla_R1 = pivot + (1.1 * range_12h / 12)

    camarilla_S1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S1)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R1)

    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches/bounces off S1 with volume spike and 12h uptrend
            if low[i] <= camarilla_S1_aligned[i] * 1.002 and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches/rejects R1 with volume spike and 12h downtrend
            elif high[i] >= camarilla_R1_aligned[i] * 0.998 and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches midpoint or 12h trend turns down
            midpoint = (camarilla_S1_aligned[i] + camarilla_R1_aligned[i]) / 2.0
            if close[i] >= midpoint or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches midpoint or 12h trend turns up
            midpoint = (camarilla_S1_aligned[i] + camarilla_R1_aligned[i]) / 2.0
            if close[i] <= midpoint or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals