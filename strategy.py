#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Use Camarilla pivot levels (R1/S1) on 1d for entry, confirmed by 1d EMA50 trend and volume spikes (>1.5x 20-period average). Enter long on break above R1 with bullish trend and volume; short on break below S1 with bearish trend and volume. Exit on close back inside the (S1,R1) range. Designed for 4h timeframe to target 25-40 trades/year, combining mean-reversion levels with trend filter to work in both bull and bear markets.

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

    # Get 1d data for Camarilla pivots, EMA trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate Camarilla levels for 1d (using previous day's range)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    range_1d = high_1d - low_1d
    r1 = close_1d + 1.1 * range_1d / 12
    s1 = close_1d - 1.1 * range_1d / 12
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    r1_shifted[0] = r1[0]  # first day uses its own level
    s1_shifted[0] = s1[0]

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average (1d volume for signal)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_shifted[i]) or np.isnan(s1_shifted[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with bullish trend (price > EMA50) and volume spike
            if (close[i] > r1_shifted[i] and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with bearish trend (price < EMA50) and volume spike
            elif (close[i] < s1_shifted[i] and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price closes back inside S1-R1 range
            if close[i] < r1_shifted[i] and close[i] > s1_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price closes back inside S1-R1 range
            if close[i] < r1_shifted[i] and close[i] > s1_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals