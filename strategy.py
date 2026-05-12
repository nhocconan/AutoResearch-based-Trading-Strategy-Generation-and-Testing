#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla pivot levels (R3/S3) from 1-day combined with 1-day EMA trend filter and volume confirmation.
# Breakouts at R3 (long) or S3 (short) capture momentum in trending markets, while the 1-day EMA ensures we trade
# with the higher timeframe trend. Volume filter reduces false breakouts. Designed for ~15-25 trades/year on 12h timeframe.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeS"
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

    # Get 1d data for Camarilla pivot calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1 / 2
    # S3 = Pivot - Range * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1 / 2.0
    s3_1d = pivot_1d - range_1d * 1.1 / 2.0

    # Align Camarilla levels to 12h timeframe (using previous day's values)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)

    # Get 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 2.0x 30-period SMA on 12h
    volume_series = pd.Series(volume)
    volume_sma30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_threshold = volume_sma30 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after indicators need 30 bars
        # Skip if any required data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + volume + 1d uptrend (price > EMA34)
            if (close[i] > r3_1d_aligned[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume + 1d downtrend (price < EMA34)
            elif (close[i] < s3_1d_aligned[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below S3 or 1d trend turns down
            if close[i] < s3_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R3 or 1d trend turns up
            if close[i] > r3_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals