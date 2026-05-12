#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1DTrend_VolumeS
# Hypothesis: Breakouts at daily Camarilla R1/S1 levels with volume confirmation and 1d trend filter.
# Uses 12h timeframe for moderate trade frequency (target 12-37/year) to minimize fee drag.
# Long: Close > daily R1 + volume > 1.5x SMA20 + price > daily EMA50
# Short: Close < daily S1 + volume > 1.5x SMA20 + price < daily EMA50
# Exit: Close crosses opposite daily Camarilla level (S1 for long, R1 for short)

name = "12h_Camarilla_R1_S1_Breakout_1DTrend_VolumeS"
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

    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Camarilla levels from previous daily close
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 12h bar
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)[i]
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)[i]
        ema50_aligned = ema50_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_aligned) or np.isnan(s1_aligned) or 
            np.isnan(ema50_aligned) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above daily R1 + volume spike (1.5x) + daily uptrend
            if (close[i] > r1_aligned and
                volume[i] > volume_threshold[i] and
                close[i] > ema50_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below daily S1 + volume spike (1.5x) + daily downtrend
            elif (close[i] < s1_aligned and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below daily S1
            if close[i] < s1_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above daily R1
            if close[i] > r1_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals