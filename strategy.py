#!/usr/bin/env python3
"""
4h_Pivot_R3S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R3/S3) from 1d act as strong support/resistance. Breakouts with volume confirmation and 1d trend filter capture strong directional moves. Works in both bull and bear markets by trading breakouts in the direction of the 1d trend. Low trade frequency avoids fee drag.
"""

name = "4h_Pivot_R3S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for 1d
    # R4 = C + ((H-L)*1.1/2)*1.1
    # R3 = C + ((H-L)*1.1/4)*1.1
    # S3 = C - ((H-L)*1.1/4)*1.1
    # S4 = C - ((H-L)*1.1/2)*1.1
    rng = high_1d - low_1d
    r3 = close_1d + (rng * 1.1 / 4) * 1.1
    s3 = close_1d - (rng * 1.1 / 4) * 1.1

    # Align R3/S3 to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R3 + volume spike + 1d uptrend
            if close[i] > r3_aligned[i] and volume[i] > vol_avg_20[i] * 1.5 and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S3 + volume spike + 1d downtrend
            elif close[i] < s3_aligned[i] and volume[i] > vol_avg_20[i] * 1.5 and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S3 or 1d trend turns down
            if close[i] < s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R3 or 1d trend turns up
            if close[i] > r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals