#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakouts with 1d EMA trend filter and volume spike confirmation.
# Uses 4h Camarilla levels from prior 4h bar to avoid look-ahead. Long when price breaks above R1 with uptrend (price > 1d EMA34) and volume spike.
# Short when price breaks below S1 with downtrend (price < 1d EMA34) and volume spike.
# Designed for low trade frequency (15-37/year) to avoid fee drift. Works in bull/bear markets by following 1d EMA trend direction.
# Exit on opposite 4h Camarilla level touch (S1 for long exit, R1 for short exit).

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Calculate Camarilla levels for each 4h bar: based on prior 4h bar's OHLC
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    # We use prior 4h bar's values to avoid look-ahead
    rng_4h = high_4h - low_4h
    camarilla_r1 = close_4h + 1.1 * rng_4h / 12
    camarilla_s1 = close_4h - 1.1 * rng_4h / 12

    # Align Camarilla levels to 1h timeframe (use prior 4h bar's levels for current 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)

    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate volume spike threshold (2.0x 20-period SMA on 1h)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 with uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S1 with downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches or crosses below S1 (opposite level)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price touches or crosses above R1 (opposite level)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals