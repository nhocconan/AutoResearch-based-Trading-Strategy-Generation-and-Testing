#!/usr/bin/env python3
"""
12h_TrueRange_Reversal_Bollinger_Trend_1d
Hypothesis: Price reversing from Bollinger Bands (20,2) extremes on 12h with strong 1d trend (EMA50) and above-average volume captures mean-reversion moves in ranging markets and trend continuations in trending markets. Works in both bull/bear by filtering with 1d EMA50 direction.
"""

name = "12h_TrueRange_Reversal_Bollinger_Trend_1d"
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

    # 12h Bollinger Bands (20,2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2 * dev
    lower_band = basis - 2 * dev

    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume filter: > 1.3x 20-period average (12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after EMA50 warmup
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at or below lower Bollinger Band + 1d EMA50 uptrend + volume
            if (close[i] <= lower_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at or above upper Bollinger Band + 1d EMA50 downtrend + volume
            elif (close[i] >= upper_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back to basis (mean reversion) or trend fails
            if close[i] >= basis[i] or close[i] <= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back to basis or trend fails
            if close[i] <= basis[i] or close[i] >= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals