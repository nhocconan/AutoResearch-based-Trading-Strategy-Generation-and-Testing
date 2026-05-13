#/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: Use Camarilla pivot levels (R1/S1) from daily timeframe with 12h price breakouts.
# Enter long when price breaks above R1 with 12h uptrend and volume spike.
# Enter short when price breaks below S1 with 12h downtrend and volume spike.
# Exit when price returns to the Camarilla pivot point (central level).
# Uses 12h timeframe to reduce trade frequency and avoid fee drag.
# Works in both bull and bear markets by following 12h trend via price vs pivot.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
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

    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day
    # Using formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Pivot = (high + low + close)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Shift by 1 to use previous day's data (no look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan

    # Calculate Camarilla levels
    R1 = close_1d_prev + 1.1 * (high_1d_prev - low_1d_prev) / 12
    S1 = close_1d_prev - 1.1 * (high_1d_prev - low_1d_prev) / 12
    Pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3

    # Align to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(Pivot_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + volume spike
            if close[i] > R1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike
            elif close[i] < S1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point
            if close[i] <= Pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point
            if close[i] >= Pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals