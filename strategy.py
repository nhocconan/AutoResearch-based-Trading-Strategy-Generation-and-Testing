#!/usr/bin/env python3
# 4h_Pivot_Touch_Trend_Volume
# Hypothesis: Price touching Camarilla pivot levels (S1/S3 for long, R1/R3 for short) combined with 1d trend filter (EMA50) and volume spikes (>2x 20-period average) provides high-probability entries in both bull and bear markets. Pivot levels act as support/resistance in ranging markets and breakout triggers in trending markets. Targets 20-50 trades/year to minimize fee drag.

name = "4h_Pivot_Touch_Trend_Volume"
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

    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels from previous day
    # CP = (High + Low + Close) / 3
    # Range = High - Low
    cp = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d

    # Camarilla levels
    # S1 = CP - (Range * 1.1 / 12)
    # S2 = CP - (Range * 1.1 / 6)
    # S3 = CP - (Range * 1.1 / 4)
    # R1 = CP + (Range * 1.1 / 12)
    # R2 = CP + (Range * 1.1 / 6)
    # R3 = CP + (Range * 1.1 / 4)
    s1 = cp - (rng * 1.1 / 12.0)
    s3 = cp - (rng * 1.1 / 4.0)
    r1 = cp + (rng * 1.1 / 12.0)
    r3 = cp + (rng * 1.1 / 4.0)

    # Align pivot levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if any required value is NaN
        if (np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches S1 or S3 (support) + price > 1d EMA50 + volume spike
            if ((low[i] <= s1_aligned[i] or low[i] <= s3_aligned[i]) and
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 or R3 (resistance) + price < 1d EMA50 + volume spike
            elif ((high[i] >= r1_aligned[i] or high[i] >= r3_aligned[i]) and
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches R1 or R3 (resistance) or trend turns bearish
            if (high[i] >= r1_aligned[i] or high[i] >= r3_aligned[i] or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches S1 or S3 (support) or trend turns bullish
            if (low[i] <= s1_aligned[i] or low[i] <= s3_aligned[i] or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals