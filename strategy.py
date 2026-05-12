#!/usr/bin/env python3
# 4h_TRIX_1dTrend_VolumeSpike
# Hypothesis: TRIX crossing zero with 1-day EMA trend filter and volume confirmation captures momentum shifts in both bull and bear markets. The 1-day EMA provides higher timeframe trend context, while volume confirmation filters false signals. Designed for 4h timeframe to limit trade frequency and reduce fee drag.

name = "4h_TRIX_1dTrend_VolumeSpike"
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

    # Calculate 1-day EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate TRIX (15-period)
    ema1 = pd.Series(close).ewm(span=15, adjust=False).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values

    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(trix[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + 1d EMA34 uptrend + volume confirmation
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + 1d EMA34 downtrend + volume confirmation
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals