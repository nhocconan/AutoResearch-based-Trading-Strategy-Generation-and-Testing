#!/usr/bin/env python3
# 6h_Russell2000_EMA_Trend_Momentum
# Hypothesis: Russell 2000 index momentum (via 10-day ROC) combined with EMA trend filter
# on 6x timeframe. Long when ROC > 0 and price > EMA20, short when ROC < 0 and price < EMA20.
# Uses 1d EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Volume confirmation ensures institutional participation. Designed for low turnover.

name = "6h_Russell2000_EMA_Trend_Momentum"
timeframe = "6h"
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

    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # EMA20 on 6x
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    # 10-period ROC on 6x (momentum)
    roc = np.zeros_like(close)
    roc[10:] = (close[10:] - close[:-10]) / close[:-10] * 100

    # Volume spike: current > 1.8x average of last 8 bars (1.33 days)
    vol_ma = pd.Series(volume).rolling(window=8, min_periods=8).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(roc[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: ROC > 0 + price > EMA20 + 1d EMA50 uptrend + volume spike
            if (roc[i] > 0 and 
                close[i] > ema_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: ROC < 0 + price < EMA20 + 1d EMA50 downtrend + volume spike
            elif (roc[i] < 0 and 
                  close[i] < ema_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ROC crosses below zero or trend breaks
            if roc[i] < 0 or close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ROC crosses above zero or trend breaks
            if roc[i] > 0 or close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals