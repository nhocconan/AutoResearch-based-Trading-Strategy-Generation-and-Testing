#!/usr/bin/env python3
# 1d_TRIX_Volume_Spike
# Hypothesis: TRIX momentum with volume spikes and 1-week trend filter.
# TRIX crossing above/below zero indicates momentum shift. Volume spike confirms strength.
# 1-week EMA provides trend filter to avoid counter-trend trades.
# Designed for low trade frequency (10-25/year) to minimize fee drag on 1d timeframe.
# Works in bull/bear markets by aligning with higher timeframe trend.

name = "1d_TRIX_Volume_Spike"
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
    volume = prices['volume'].values

    # Get 1d data for TRIX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate TRIX (1-period rate of change of triple EMA)
    # TRIX = [(EMA3 - EMA3_prev) / EMA3_prev] * 100
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = (np.diff(ema3, prepend=ema3[0]) / np.where(ema3 == 0, 1e-10, ema3)) * 100

    # Align TRIX to 1d timeframe (same timeframe, so no alignment needed)
    trix_aligned = trix  # Already on 1d timeframe

    # Get 1 week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + price above 1w EMA50 + volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and
                close[i] > ema50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + price below 1w EMA50 + volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and
                  close[i] < ema50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals