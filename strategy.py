#!/usr/bin/env python3
# 4h_12h_Camarilla_R3_S3_Breakout_Volume
# Hypothesis: Breakouts at 12h Camarilla R3/S3 levels with volume confirmation on 4h timeframe.
# Uses 12h timeframe for Camarilla levels and 4h for entry/exit, designed to work in both bull and bear markets.
# Targets 20-50 trades/year on 4h timeframe to avoid excessive trading and fee drag.
# Volume confirmation requires current volume > 1.5x the average of the last 10 periods.

name = "4h_12h_Camarilla_R3_S3_Breakout_Volume"
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

    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)

    # Calculate 12h close for Camarilla calculation
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values

    # Calculate Camarilla R3 and S3 levels from previous 12h OHLC
    prev_close = close_12h[1:]  # shift(1) equivalent
    prev_high = high_12h[1:]
    prev_low = low_12h[1:]
    # Pad with NaN for the first element
    prev_close = np.concatenate([[np.nan], prev_close])
    prev_high = np.concatenate([[np.nan], prev_high])
    prev_low = np.concatenate([[np.nan], prev_low])

    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4

    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)

    # Volume confirmation: current volume > 1.5x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Camarilla R3 with volume confirmation
            if close[i] > camarilla_r3_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S3 with volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R3
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S3
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals