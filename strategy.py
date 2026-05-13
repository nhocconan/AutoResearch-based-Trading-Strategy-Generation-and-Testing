#!/usr/bin/env python3
# 6h_PivotReversal_VolumeFilter
# Hypothesis: Price rejection at daily pivot support/resistance levels with volume exhaustion
# captures reversals in both trending and ranging markets. Uses daily pivot points (PP, R1, S1)
# as key institutional levels. Long when price bounces off S1 with declining volume;
# short when price reverses from R1 with declining volume. Avoids whipsaws by requiring
# volume to be below average, indicating exhaustion of the prior move.
# Target: 15-25 trades/year on 6h to stay within optimal range.

name = "6h_PivotReversal_VolumeFilter"
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

    # Get daily pivot points (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate daily pivot: PP = (H + L + C)/3
    # R1 = 2*PP - L, S1 = 2*PP - H
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    r1 = 2 * pp - df_1d['low']
    s1 = 2 * pp - df_1d['high']

    # Align to 6h timeframe (use prior day's pivot for current day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)

    # Volume filter: volume < 0.8 * 20-period average (exhaustion signal)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at or above S1 with volume exhaustion (rejection of lows)
            if (close[i] >= s1_aligned[i] * 0.999 and  # allow small buffer
                volume[i] < vol_avg_20[i] * 0.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or below R1 with volume exhaustion (rejection of highs)
            elif (close[i] <= r1_aligned[i] * 1.001 and  # allow small buffer
                  volume[i] < vol_avg_20[i] * 0.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot point or volume picks up (failure)
            if (close[i] >= pp_aligned[i] or 
                volume[i] > vol_avg_20[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot point or volume picks up (failure)
            if (close[i] <= pp_aligned[i] or 
                volume[i] > vol_avg_20[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals