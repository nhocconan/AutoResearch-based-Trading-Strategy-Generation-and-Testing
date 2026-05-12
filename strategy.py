#!/usr/bin/env python3
# 1d_WeeklyPivot_Breakout_TrendVolume_v1
# Hypothesis: 1d price breaks above/below previous week's pivot levels with 1w trend filter and volume confirmation.
# Uses weekly pivot points (PP, R1, S1) for key support/resistance, 1w EMA34 for trend direction,
# and volume spike (1.8x 20-day average) to confirm breakout strength.
# Designed for 10-20 trades/year to avoid fee drag. Works in bull/bear markets by following 1w trend.

name = "1d_WeeklyPivot_Breakout_TrendVolume_v1"
timeframe = "1d"
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

    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points (PP, R1, S1) from previous week
    pivot_pp = (high_1w + low_1w + close_1w) / 3.0
    pivot_r1 = 2 * pivot_pp - low_1w
    pivot_s1 = 2 * pivot_pp - high_1w

    # Align weekly pivots to daily timeframe (available after weekly close)
    pivot_pp_aligned = align_htf_to_ltf(prices, df_1w, pivot_pp)
    pivot_r1_aligned = align_htf_to_ltf(prices, df_1w, pivot_r1)
    pivot_s1_aligned = align_htf_to_ltf(prices, df_1w, pivot_s1)

    # Get 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate daily volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.8  # Require 1.8x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_pp_aligned[i]) or np.isnan(pivot_r1_aligned[i]) or 
            np.isnan(pivot_s1_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above weekly R1 in 1w uptrend with volume spike
            if (close[i] > pivot_r1_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S1 in 1w downtrend with volume spike
            elif (close[i] < pivot_s1_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly pivot point (trend reversal)
            if close[i] < pivot_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly pivot point (trend reversal)
            if close[i] > pivot_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals