#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Price breaking above/below daily Camarilla R3/S3 levels with 1-day trend filter and volume confirmation captures strong trending moves. Works in bull/bear by following daily trend direction. Uses 12h timeframe to reduce trade frequency and avoid fee drag, with daily timeframe for trend context and breakout levels.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate daily high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels: R3, S3
    camarilla_range = high_1d - low_1d
    r3_level = close_1d + 1.1 * camarilla_range / 2
    s3_level = close_1d - 1.1 * camarilla_range / 2

    # Align Camarilla levels to 12h timeframe
    r3_level_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_level_aligned = align_htf_to_ltf(prices, df_1d, s3_level)

    # 1d trend filter: price above/below close (simple trend)
    trend_1d = close_1d  # Using close as trend proxy
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after volume MA warmup
        if (np.isnan(r3_level_aligned[i]) or np.isnan(s3_level_aligned[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 + price above daily close + volume confirmation
            if (close[i] > r3_level_aligned[i] and 
                close[i] > trend_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + price below daily close + volume confirmation
            elif (close[i] < s3_level_aligned[i] and 
                  close[i] < trend_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below daily close (trend reversal)
            if close[i] < trend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above daily close (trend reversal)
            if close[i] > trend_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals