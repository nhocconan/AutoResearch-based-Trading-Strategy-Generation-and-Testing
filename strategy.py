#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS_v3
# Hypothesis: Optimized Camarilla R1/S1 breakout with volume confirmation and 1d EMA trend filter.
# Uses tighter volume threshold (3x SMA20) and requires price close beyond level to reduce false breakouts.
# Designed for low trade frequency (target 20-30/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
# Trend filter uses 1d EMA34 to ensure alignment with higher timeframe direction.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS_v3"
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

    # Get 1d data for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Camarilla levels from previous 1d close
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12

    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: 3x 20-period SMA (tighter to reduce trades)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 3.0  # Increased from 2x to 3x for fewer false signals

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 4h bar
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)[i]
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)[i]
        ema34_aligned = ema34_1d_aligned[i]

        # Skip if any required data is NaN
        if (np.isnan(r1_aligned) or np.isnan(s1_aligned) or 
            np.isnan(ema34_aligned) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price closes above Camarilla R1 + volume spike (3x) + 1d uptrend
            if (close[i] > r1_aligned and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_aligned):
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below Camarilla S1 + volume spike (3x) + 1d downtrend
            elif (close[i] < s1_aligned and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_aligned):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 OR 1d trend turns down
            if close[i] < s1_aligned or close[i] < ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 OR 1d trend turns up
            if close[i] > r1_aligned or close[i] > ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals