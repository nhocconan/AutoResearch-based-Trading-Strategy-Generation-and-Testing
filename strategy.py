#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot breakouts with daily trend and volume filter capture momentum in both bull and bear markets.
# Uses 1d EMA50 for trend direction and 1h for entry timing. Session filter (08-20 UTC) reduces noise.
# Target: 15-30 trades/year on 1h to avoid fee drag.

name = "1h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "1h"
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

    # Get 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Camarilla levels from previous day (using prior day's close)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Calculate Camarilla for each 1d bar using previous day's data
    camarilla_r1 = np.zeros(len(close_1d))
    camarilla_s1 = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        camarilla_r1[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
        camarilla_s1[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume confirmation: volume > 1.5x 24-period average
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        hour = hours[i]
        in_session = (8 <= hour <= 20)

        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + 1d uptrend + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Close below S1 + 1d downtrend + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_24[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to S1 or trend change
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Mean reversion to R1 or trend change
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals