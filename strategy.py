#!/usr/bin/env python3
# 12h_TRIX_ZeroCross_1dTrend_Volume
# Hypothesis: TRIX zero-cross on 12h with 1d trend filter and volume confirmation.
# TRIX (12-period) captures momentum shifts; zero-cross indicates trend changes.
# Combined with 1d EMA34 trend filter to ensure alignment with higher timeframe trend.
# Volume spike confirms institutional participation. Works in bull via long entries in uptrends,
# and in bear via short entries in downtrends. Target: 12-37 trades/year.

name = "12h_TRIX_ZeroCross_1dTrend_Volume"
timeframe = "12h"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # TRIX (12,12,12) on 12h
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value has no previous

    # Volume spike: current > 2.0x average of last 12 bars (6 days)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(12, n):  # Start after TRIX warmup
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + 1d EMA34 uptrend + volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + 1d EMA34 downtrend + volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or trend breaks
            if (trix[i] < 0 and trix[i-1] >= 0) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or trend breaks
            if (trix[i] > 0 and trix[i-1] <= 0) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals