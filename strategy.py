#!/usr/bin/env python3
# 1d_TRIX_ZeroCross_VolumeFilter
# Hypothesis: TRIX zero-line crossovers indicate trend changes. Long when TRIX crosses above zero, short when below.
# Filtered by volume spikes (2x 20-bar average) to ensure institutional participation.
# Works in bull markets via long entries in uptrends, and in bear markets via short entries in downtrends.
# Target: 10-20 trades/year on daily timeframe.

name = "1d_TRIX_ZeroCross_VolumeFilter"
timeframe = "1d"
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

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # TRIX calculation (15-period) on daily
    # EMA1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA2
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    # EMA3
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # TRIX = 100 * (EMA3 - prev EMA3) / prev EMA3
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # first value undefined
    trix = trix_raw  # already the final indicator

    # Volume spike: current > 2.0x average of last 20 bars
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):  # Start after TRIX warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + 1w EMA50 uptrend + volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + 1w EMA50 downtrend + volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend breaks
            if trix[i] < 0 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend breaks
            if trix[i] > 0 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals