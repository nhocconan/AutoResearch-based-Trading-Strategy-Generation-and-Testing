#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as strong support/resistance.
# Breakout above R1 or below S1 with volume confirmation and in direction of daily EMA34 trend
# captures momentum moves. Works in both bull and bear markets by trading with the daily trend.
# Target: 20-40 trades/year per symbol.

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

    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0

    # Daily trend: EMA34 on close
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_uptrend = close_1d > ema34
    daily_downtrend = close_1d < ema34

    # Volume confirmation: current 4h volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    # Align daily levels to 4h timeframe (wait for daily bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after warmup for EMA34
        # Skip if any aligned values are NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Daily uptrend + price breaks above R1 + volume spike
            if (daily_uptrend_aligned[i] > 0.5 and 
                close[i] > r1_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Daily downtrend + price breaks below S1 + volume spike
            elif (daily_downtrend_aligned[i] > 0.5 and 
                  close[i] < s1_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 or trend changes
            if (close[i] < s1_aligned[i] or daily_uptrend_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 or trend changes
            if (close[i] > r1_aligned[i] or daily_downtrend_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals