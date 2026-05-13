#!/usr/bin/env python3
# 6h_Williams_Alligator_1wTrend_VolumeFilter
# Hypothesis: Williams Alligator (13,8,5 SMAs) on 1w defines trend. On 6s, enter long when green line > red line > blue line with volume spike; short when reverse. Exit when lines re-intertwine. Uses weekly trend filter to avoid counter-trend trades in 6s timeframe. Designed for trending markets with noise filtering via volume. Target: 15-35 trades/year per symbol.

name = "6h_Williams_Alligator_1wTrend_VolumeFilter"
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

    # Get 1w data for Williams Alligator (13,8,5 SMAs on median price)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    median_1w = (high_1w + low_1w) / 2.0

    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) SMAs of median
    jaw = pd.Series(median_1w).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_1w).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_1w).rolling(window=5, min_periods=5).mean().values

    # Align to 6s timeframe (weekly trend only updates after weekly close)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: lips > teeth > jaw (bullish alignment) + volume spike
            if lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: jaws > teeth > lips (bearish alignment) + volume spike
            elif jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: when alligator lines re-intertwine (not perfect order)
            if not (lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: when alligator lines re-intertwine
            if not (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals