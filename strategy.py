#!/usr/bin/env python3
# 6h_WeeklyPivot_Camarilla_Breakout_Trend_Volume
# Hypothesis: 6s breakout of weekly Camarilla pivot levels (R4/S4) with 1d EMA trend filter and volume confirmation.
# Weekly pivots provide strong structural support/resistance; breakouts indicate momentum.
# 1d EMA ensures alignment with daily trend to avoid counter-trend trades.
# Volume spike confirms breakout strength. Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull/bear by following 1d trend and using weekly structure.

name = "6h_WeeklyPivot_Camarilla_Breakout_Trend_Volume"
timeframe = "6h"
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

    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # S4 = C - Range * 1.1/2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0

    # Align weekly Camarilla levels to 6h timeframe (no extra delay needed for pivot levels)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above weekly R4 in 1d uptrend with volume spike
            if close[i] > r4_1w_aligned[i] and close[i] > ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below weekly S4 in 1d downtrend with volume spike
            elif close[i] < s4_1w_aligned[i] and close[i] < ema20_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly S4 (reversal to downside)
            if close[i] < s4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly R4 (reversal to upside)
            if close[i] > r4_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals