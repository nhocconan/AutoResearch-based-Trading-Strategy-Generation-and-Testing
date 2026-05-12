#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2
# Hypothesis: Camarilla pivot R3/S3 breakouts with 1d EMA34 trend filter and volume spike
# capture institutional breakouts in both bull and bear markets. The Camarilla levels
# act as institutional reference points, EMA34 ensures alignment with daily trend,
# and volume confirmation filters false breakouts. Designed for ~20-30 trades/year.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v2"
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

    # Get 1d data for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    # Calculate Camarilla pivot levels from previous day
    # R3 = close + 1.1*(high-low)*1.1/2, S3 = close - 1.1*(high-low)*1.1/2
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 2.0x 24-period SMA (1 day of 4h bars)
    volume_series = pd.Series(volume)
    volume_sma24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_threshold = volume_sma24 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after indicators need 34 bars
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3 + volume + 1d uptrend
            if (close[i] > camarilla_r3_aligned[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 + volume + 1d downtrend
            elif (close[i] < camarilla_s3_aligned[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S3 OR 1d trend turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R3 OR 1d trend turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals