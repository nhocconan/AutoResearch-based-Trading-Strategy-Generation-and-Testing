#!/usr/bin/env python3
# 6h_PivotReversal_VolumeSpike
# Hypothesis: In 6h timeframe, price reversals at daily pivot support/resistance levels
# with volume spike confirmation capture mean-reversion moves in ranging markets
# and continuation moves in trending markets. Works in both bull and bear regimes
# by fading extremes and catching breakouts with volume confirmation.
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.

name = "6h_PivotReversal_VolumeSpike"
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

    # Get daily data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate daily pivot points (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)

    # Align daily pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume spike: current volume > 2.0 x 20-period average (higher threshold for fewer trades)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price near S1/S2 with volume spike (mean reversion bounce)
            if ((close[i] <= s1_aligned[i] * 1.005 or close[i] <= s2_aligned[i] * 1.005) and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price near R1/R2 with volume spike (mean reversion fade)
            elif ((close[i] >= r1_aligned[i] * 0.995 or close[i] >= r2_aligned[i] * 0.995) and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot or R1, or volume spikes at resistance (failure)
            if (close[i] >= pivot_aligned[i] or close[i] >= r1_aligned[i] or 
                (close[i] >= r1_aligned[i] * 0.995 and volume_spike[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot or S1, or volume spikes at support (failure)
            if (close[i] <= pivot_aligned[i] or close[i] <= s1_aligned[i] or 
                (close[i] <= s1_aligned[i] * 1.005 and volume_spike[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals