#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot R3/S3 breakout on 12h with daily EMA34 trend filter and volume spike confirmation.
# Works in bull markets by buying R3 breakouts in uptrend, in bear markets by selling S3 breakdowns in downtrend.
# Uses volume spike to avoid false breakouts. Targets low trade frequency for 12h timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)

    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values

    # Calculate Camarilla levels for each 12h bar
    # R4 = close + (high - low) * 1.1/2, R3 = close + (high - low) * 1.1/4, etc.
    # S3 = close - (high - low) * 1.1/4, S4 = close - (high - low) * 1.1/2
    hl_range = high_12h - low_12h
    r3_12h = close_12h + hl_range * 1.1 / 4
    s3_12h = close_12h - hl_range * 1.1 / 4

    # Align Camarilla levels to 12h timeframe (no additional delay needed)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)

    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate volume spike threshold (2.0x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R3 in uptrend with volume spike
            if (close[i] > r3_12h_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3 in downtrend with volume spike
            elif (close[i] < s3_12h_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close falls below S3 (reversal signal)
            if close[i] < s3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close rises above R3 (reversal signal)
            if close[i] > r3_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals