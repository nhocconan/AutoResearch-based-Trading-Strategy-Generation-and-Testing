#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_Volume
Hypothesis: Breakouts at daily Camarilla R3/S3 levels with volume confirmation and 12h trend filter.
Uses 6h timeframe to target ~15-25 trades/year. R3/S3 levels are stronger than R1/S1, reducing false breakouts.
12h trend filter helps avoid counter-trend trades in choppy markets. Volume confirmation ensures breakout strength.
Designed to work in both bull and bear regimes by following the 12h trend direction.
"""

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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

    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate daily Camarilla levels (R3/S3 are stronger breakout levels)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + camarilla_range * 1.1 / 4
    s3 = close_1d - camarilla_range * 1.1 / 4

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)

    close_12h = df_12h['close'].values
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Volume confirmation: 2x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 2.0

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Get aligned values for current 6h bar (only uses completed higher timeframe bars)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)[i]
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)[i]
        ema50_aligned = ema50_12h_aligned[i]
        vol_threshold_val = volume_threshold[i]

        # Skip if any required data is NaN
        if (np.isnan(r3_aligned) or np.isnan(s3_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(vol_threshold_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above daily R3 + volume spike (2x) + 12h uptrend
            if (close[i] > r3_aligned and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below daily S3 + volume spike (2x) + 12h downtrend
            elif (close[i] < s3_aligned and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below daily S3 (reversal signal)
            if close[i] < s3_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above daily R3 (reversal signal)
            if close[i] > r3_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals