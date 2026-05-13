#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Reversal_With_Volume_Confirmation
# Hypothesis: Price reversals at Camarilla pivot levels (S3/S4 for long, R3/R4 for short) provide
# high-probability setups when aligned with daily trend and confirmed by volume spikes.
# Works in both bull and bear markets by following the daily trend direction.
# Targets low-frequency, high-quality setups to minimize fee drag.

name = "4h_Camarilla_Pivot_Reversal_With_Volume_Confirmation"
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

    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # S3 = C - (Range * 1.1 / 4)
    # S4 = C - (Range * 1.1 / 2)
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # R3 = C + (Range * 1.1 / 4)
    # R4 = C + (Range * 1.1 / 2)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    s3 = close_1d - (rng * 1.1 / 4.0)
    s4 = close_1d - (rng * 1.1 / 2.0)
    r3 = close_1d + (rng * 1.1 / 4.0)
    r4 = close_1d + (rng * 1.1 / 2.0)

    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)

    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume spike: volume > 2.0 * 20-period average (~5 days at 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + price at S3/S4 support + volume spike
            if close[i] > ema34_aligned[i] and (close[i] <= s3_aligned[i] * 1.001 or close[i] <= s4_aligned[i] * 1.001) and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + price at R3/R4 resistance + volume spike
            elif close[i] < ema34_aligned[i] and (close[i] >= r3_aligned[i] * 0.999 or close[i] >= r4_aligned[i] * 0.999) and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R3/R4 or trend turns bearish
            if close[i] >= r3_aligned[i] * 0.999 or close[i] >= r4_aligned[i] * 0.999 or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S3/S4 or trend turns bullish
            if close[i] <= s3_aligned[i] * 1.001 or close[i] <= s4_aligned[i] * 1.001 or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals