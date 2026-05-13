#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: Breakout at Camarilla R3 (resistance) or S3 (support) with volume spike and 12h EMA50 trend filter.
# Camarilla levels provide high-probability reversal/breakout points. Volume surge confirms institutional participation.
# Trend filter ensures alignment with higher timeframe momentum, reducing false signals in choppy markets.
# Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend).
# Low frequency due to specific level breaks and strict volume confirmation.

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + (H - L) * 1.1 / 2
    # S3 = Pivot - (H - L) * 1.1 / 2
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d
    r3 = pivot + (rng * 1.1 / 2.0)
    s3 = pivot - (rng * 1.1 / 2.0)

    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)

    # Volume spike: volume > 2.0 * 6-period average (1 day worth at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R3 + 12h uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S3 + 12h downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S3 or trend reversal
            if close[i] < s3_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R3 or trend reversal
            if close[i] > r3_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals