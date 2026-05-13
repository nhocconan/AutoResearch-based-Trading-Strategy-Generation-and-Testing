#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Pullback_With_Volume
Hypothesis: Weekly pivot points act as strong support/resistance levels. Price pulling back to
S1/R1 after touching S2/R2 or S3/R3 with volume confirmation provides high-probability
continuation trades in the direction of the weekly trend. Works in both bull and bear markets
by aligning with weekly trend direction. Targets low-frequency, high-quality setups.
"""

name = "6h_Weekly_Pivot_Pullback_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values

    # Calculate weekly pivot points: P = (H+L+C)/3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Support and resistance levels
    s1_w = 2 * pivot_w - high_w
    r1_w = 2 * pivot_w - low_w
    s2_w = pivot_w - (high_w - low_w)
    r2_w = pivot_w + (high_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    r3_w = high_w + 2 * (pivot_w - low_w)

    # Align weekly pivot levels to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)

    # Weekly trend: price above/below 20-period EMA (approx 5 months)
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)

    # Volume spike on 6t: volume > 2.0 * 20-period average (~5 days at 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or
            np.isnan(r1_w_aligned[i]) or np.isnan(ema20_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + pullback to S1 after touching S2/S3 + volume spike
            if (close[i] > ema20_w_aligned[i] and
                low[i] <= s2_w_aligned[i] * 1.002 and  # touched S2 or lower
                close[i] >= s1_w_aligned[i] * 0.999 and  # now at S1
                close[i] <= s1_w_aligned[i] * 1.001 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + pullback to R1 after touching R2/R3 + volume spike
            elif (close[i] < ema20_w_aligned[i] and
                  high[i] >= r2_w_aligned[i] * 0.998 and  # touched R2 or higher
                  close[i] <= r1_w_aligned[i] * 1.001 and  # now at R1
                  close[i] >= r1_w_aligned[i] * 0.999 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 or trend turns bearish
            if close[i] >= r1_w_aligned[i] * 0.999 or close[i] < ema20_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 or trend turns bullish
            if close[i] <= s1_w_aligned[i] * 1.001 or close[i] > ema20_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals