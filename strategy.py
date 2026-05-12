#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Camarilla R1/S1 breakout from daily levels with 12h EMA trend filter and volume spike confirmation on 4h timeframe.
# Uses Camarilla pivot levels (R1, S1) calculated from prior day's OHLC. Long when price breaks above R1 with uptrend (price > EMA50) and volume spike.
# Short when price breaks below S1 with downtrend (price < EMA50) and volume spike. Designed for low trade frequency (19-50/year) to avoid fee drag.
# Works in bull/bear markets by following 12h EMA trend direction. Exit on opposite Camarilla level touch (S1 for long exit, R1 for short exit).

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

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each day: based on prior day's OHLC
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    # We use prior day's values to avoid look-ahead
    rng_1d = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng_1d / 12
    camarilla_s1 = close_1d - 1.1 * rng_1d / 12

    # Align Camarilla levels to 4h timeframe (use prior day's levels for current day)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Calculate volume spike threshold (2.0x 20-period SMA on 4h)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 with uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 with downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches or crosses below S1 (opposite level)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches or crosses above R1 (opposite level)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals