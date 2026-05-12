#!/usr/bin/env python3
# 1D_Camarilla_R1_S1_Breakout_1WTrend_VolumeS
# Hypothesis: Camarilla pivot R1/S1 breakouts on daily chart with weekly trend filter and volume confirmation
# work in both bull and bear markets. Camarilla levels provide institutional-grade support/resistance,
# weekly trend ensures directional alignment, volume reduces false breakouts. Targets 15-25 trades/year.

name = "1D_Camarilla_R1_S1_Breakout_1WTrend_VolumeS"
timeframe = "1d"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 21-period EMA for weekly trend
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous day's data, so shift by 1
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First value
    prev_low[0] = low[0]
    prev_close[0] = close[0]

    camarilla_range = prev_high - prev_low
    camarilla_r1 = prev_close + camarilla_range * 1.1 / 12
    camarilla_s1 = prev_close - camarilla_range * 1.1 / 12

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after indicators need 20 bars
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema21_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + volume + weekly uptrend
            if (close[i] > camarilla_r1[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 + volume + weekly downtrend
            elif (close[i] < camarilla_s1[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 OR weekly trend turns down
            if close[i] < camarilla_s1[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 OR weekly trend turns up
            if close[i] > camarilla_r1[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals