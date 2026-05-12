#!/usr/bin/env python3
# 1D_CCI_MeanReversion_1wTrend_Filter
# Hypothesis: Mean reversion using CCI(20) on daily timeframe with weekly trend filter.
# Long when CCI < -100 and weekly trend up; short when CCI > 100 and weekly trend down.
# Works in both bull and bear markets by fading extremes in ranging markets while
# respecting the higher timeframe trend. Target: 15-25 trades/year.

name = "1D_CCI_MeanReversion_1wTrend_Filter"
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

    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # CCI calculation (20-period) on daily
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 and CCI warmup
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(sma_tp[i]) or 
            np.isnan(mad[i]) or np.isnan(cci[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: CCI < -100 + weekly uptrend
            if cci[i] < -100 and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: CCI > 100 + weekly downtrend
            elif cci[i] > 100 and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: CCI crosses above -50 or trend breaks
            if cci[i] > -50 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: CCI crosses below 50 or trend breaks
            if cci[i] < 50 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals