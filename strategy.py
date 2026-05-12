#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_TrendFilter
Hypothesis: TRIX (12) crossing zero with volume spike and trend confirmation 
captures momentum bursts in both bull and bear markets. Uses 1d trend filter 
to avoid counter-trend trades. Designed for 30-50 trades/year to minimize fee drag.
"""

name = "4h_TRIX_VolumeSpike_TrendFilter"
timeframe = "4h"
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

    # Get 1d data for trend filter (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Calculate TRIX (12) on 4h close
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change()).values

    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        trix_val = trix[i]
        ema50_val = ema50_1d_aligned[i]
        vol_avg_val = vol_avg_20[i]
        vol_val = volume[i]

        if np.isnan(trix_val) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + uptrend + volume spike
            if trix_val > 0 and trix[i-1] <= 0 and close[i] > ema50_val and vol_val > vol_avg_val * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + downtrend + volume spike
            elif trix_val < 0 and trix[i-1] >= 0 and close[i] < ema50_val and vol_val > vol_avg_val * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend reversal
            if trix_val < 0 or close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend reversal
            if trix_val > 0 or close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals