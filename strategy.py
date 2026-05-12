#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R1/S1) from 4h for breakout direction, filtered by 1d EMA trend and volume spikes on 1h for entry timing.
# Camarilla levels provide precise intraday support/resistance, while 1d EMA filters counter-trend trades and volume confirms breakout strength.
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag. Works in bull/bear by following 1d trend.
# Uses 4h OHLC for Camarilla calculation and 1h volume for spike detection.

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

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate Camarilla levels for 4h: R1, S1 based on previous 4h bar
    # Camarilla formulas: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We use the previous completed 4h bar's OHLC
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = close_4h[0]  # First value
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]

    range_4h = prev_high_4h - prev_low_4h
    camarilla_r1 = prev_close_4h + range_4h * 1.1 / 12
    camarilla_s1 = prev_close_4h - range_4h * 1.1 / 12

    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)

    # Calculate 1h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Camarilla R1 in 1d uptrend with volume spike
            if close[i] > camarilla_r1_aligned[i] and close[i] > ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below Camarilla S1 in 1d downtrend with volume spike
            elif close[i] < camarilla_s1_aligned[i] and close[i] < ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 (reversal to support)
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 (reversal to resistance)
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals