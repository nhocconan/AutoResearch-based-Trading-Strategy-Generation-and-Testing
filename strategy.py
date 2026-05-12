#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Breakouts at daily Camarilla R3/S3 levels with volume confirmation and 1d trend filter on 6h timeframe.
Targets 15-35 trades/year to stay within fee limits. Uses daily for structure and trend filter.
Long: Close > daily R3 + volume > 1.5x SMA20 + close > 1d EMA50
Short: Close < daily S3 + volume > 1.5x SMA20 + close < 1d EMA50
Exit: Close crosses opposite daily Camarilla level (S3 for long, R3 for short)
"""

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Camarilla levels from previous daily close
    camarilla_range = high_1d - low_1d
    r3 = close_1d + camarilla_range * 1.1 / 4
    s3 = close_1d - camarilla_range * 1.1 / 4

    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 6h bar
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)[i]
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)[i]
        ema50_aligned = ema50_1d_aligned[i]
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
            # LONG: Price closes above daily R3 + volume spike (1.5x) + 1d uptrend
            if (close[i] > r3_aligned and
                volume[i] > vol_threshold_val and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below daily S3 + volume spike (1.5x) + 1d downtrend
            elif (close[i] < s3_aligned and
                  volume[i] > vol_threshold_val and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below daily S3
            if close[i] < s3_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above daily R3
            if close[i] > r3_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals