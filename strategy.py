#!/usr/bin/env python3
# 6h_TRIX_Momentum_1dTrend_Volume
# Hypothesis: TRIX (1-period rate of change of triple EMA) captures momentum shifts.
# Long when TRIX crosses above zero with 1d EMA uptrend and volume spike.
# Short when TRIX crosses below zero with 1d EMA downtrend and volume spike.
# Exit when TRIX crosses back across zero.
# TRIX is less noisy than MACD and works in both bull/bear via 1d trend filter.
# Target: 15-25 trades/year on 6h to minimize fee drag.

name = "6h_TRIX_Momentum_1dTrend_Volume"
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
    volume = prices['volume'].values

    # Get 1d data for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate TRIX on 1d: EMA(EMA(EMA(close, 12), 12), 12) then 1-period ROC
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = pd.Series(ema3).pct_change() * 100  # 1-period ROC as percentage
    trix = trix_raw.values

    # Align TRIX to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + 1d EMA uptrend + volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + 1d EMA downtrend + volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
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