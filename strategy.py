#!/usr/bin/env python3
# 6h_WeeklyPivot_BullBearPower_1dTrend_VolumeFilter
# Hypothesis: Use weekly pivot points (from 1w data) as structural levels and Elder Ray Bull/Bear Power (from 1d) for trend direction.
# Long when price crosses above weekly pivot R1 with Bull Power > 0 and volume spike.
# Short when price crosses below weekly pivot S1 with Bear Power < 0 and volume spike.
# Exit when price returns to the previous week's close.
# Weekly pivot provides structural support/resistance; Elder Ray filters trend; volume confirms breakout.
# Designed to work in bull (buy R1 breakouts in Bull Power>0) and bear (sell S1 breakdowns in Bear Power<0).
# Target: 15-30 trades/year per symbol.

name = "6h_WeeklyPivot_BullBearPower_1dTrend_VolumeFilter"
timeframe = "6h"
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

    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points (using previous week's data)
    # P = (H + L + C) / 3
    # R1 = 2*P - L
    # S1 = 2*P - H
    P_1w = (high_1w + low_1w + close_1w) / 3.0
    R1_1w = 2 * P_1w - low_1w
    S1_1w = 2 * P_1w - high_1w

    # Align weekly pivot levels to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    c_prev_1w = np.roll(close_1w, 1)
    c_prev_1w[0] = np.nan
    c_1w_aligned = align_htf_to_ltf(prices, df_1w, c_prev_1w)

    # Get daily data for Elder Ray and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13

    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(c_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price crosses above weekly R1 with Bull Power > 0 and volume spike
            if close[i] > r1_1w_aligned[i] and bull_power_aligned[i] > 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below weekly S1 with Bear Power < 0 and volume spike
            elif close[i] < s1_1w_aligned[i] and bear_power_aligned[i] < 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to previous week's close
            if close[i] <= c_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to previous week's close
            if close[i] >= c_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals