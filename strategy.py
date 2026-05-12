#!/usr/bin/env python3
# 4h_Trix_VolumeSpike_Trend_Filter
# Hypothesis: TRIX crossing zero with volume spike and 1-day EMA50 trend filter captures momentum shifts in both bull and bear markets. Uses 4h timeframe with daily trend filter for higher timeframe context. Low trade frequency (~20-40/year) to minimize fee drag.

name = "4h_Trix_VolumeSpike_Trend_Filter"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # TRIX: EMA(EMA(EMA(close, 12), 12), 12) - 1 period rate of change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3

    # 1-day EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: >2.0x 20-period average (strict to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + uptrend + volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: TRIX crosses below zero + downtrend + volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero (momentum fade)
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero (momentum fade)
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals