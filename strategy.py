#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla R1/S1 breakouts with 1d EMA50 trend filter and volume confirmation.
# The Camarilla levels provide institutional-grade support/resistance, while the 1d EMA50
# ensures we trade with the daily trend. Volume filter reduces false breakouts.
# Designed for ~20-40 trades/year on 1h to minimize fee drag. Works in both bull and bear
# markets by filtering breakouts with higher timeframe trend and volume confirmation.

name = "1h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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

    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Calculate Camarilla levels for each day
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (1 day = 24 hours)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 24-period SMA (approx 1 day)
    volume_series = pd.Series(volume)
    volume_sma24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_threshold = volume_sma24 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):  # Start after indicators need 24 bars
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + volume + 1d uptrend
            if (close[i] > r1_aligned[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + volume + 1d downtrend
            elif (close[i] < s1_aligned[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S1 OR 1d trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above R1 OR 1d trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals