#!/usr/bin/env python3
"""
6h_Pivot_Reversal_With_Volume_Squeeze
Hypothesis: Price rejection at 1-day pivot points (R3/S3) with volume squeeze confirmation (volume < 0.7x average) identifies exhaustion points for mean reversion. Works in both bull and bear markets by fading extremes at key support/resistance levels. Uses 60-minute EMA20 for trend filter to avoid counter-trend trades in strong trends.
"""

name = "6h_Pivot_Reversal_With_Volume_Squeeze"
timeframe = "6h"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate pivot points from 1d data
    # Standard pivot: P = (H + L + C) / 3
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)

    # Shift by 1 to use previous day's pivot levels
    prev_r3 = np.roll(r3_1d, 1)
    prev_s3 = np.roll(s3_1d, 1)
    prev_r3[0] = np.nan
    prev_s3[0] = np.nan

    # Align pivot levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, prev_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, prev_s3)

    # Volume squeeze: volume < 0.7x 20-period average (6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_squeeze = volume < (0.7 * vol_ma)

    # 60-minute EMA20 trend filter (from 1h data)
    df_1h = get_htf_data(prices, '1h')
    close_1h = df_1h['close'].values
    ema_20_1h = pd.Series(close_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_20_1h_aligned[i]) or np.isnan(volume_squeeze[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price rejects below S3 (bounces up) with volume squeeze + above EMA20
            if (low[i] < s3_aligned[i] and close[i] > s3_aligned[i] and 
                close[i] > ema_20_1h_aligned[i] and volume_squeeze[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects above R3 (falls down) with volume squeeze + below EMA20
            elif (high[i] > r3_aligned[i] and close[i] < r3_aligned[i] and 
                  close[i] < ema_20_1h_aligned[i] and volume_squeeze[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes above R3 (breakout) or volume expansion
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes below S3 (breakdown) or volume expansion
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals