#!/usr/bin/env python3
# 4h_ETF_Momentum_With_Volume_Filter
# Hypothesis: Combines ETF (Ehlers Trend Follower) for trend direction on 4h with volume confirmation (>1.5x 20-bar average) and 1d trend filter (price > 1d EMA50 for longs, < for shorts). ETF reduces lag vs traditional moving averages while maintaining smoothness. Volume filter ensures momentum is supported by participation. Targets 25-40 trades/year to avoid fee drag, works in bull/bear via 1d trend filter.

name = "4h_ETF_Momentum_With_Volume_Filter"
timeframe = "4h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # ETF (Ehlers Trend Follower) - reduces lag while smoothing
    # Alpha = 1 / (1 + sqrt(2)) for critical damping
    alpha = 1.0 / (1.0 + np.sqrt(2.0))
    etf = np.zeros(n)
    etf[0] = close[0]
    for i in range(1, n):
        etf[i] = alpha * close[i] + (1 - alpha) * etf[i-1]

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        # Skip if any required value is NaN
        if (np.isnan(etf[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > ETF (bullish) + price > 1d EMA50 + volume confirmation
            if (close[i] > etf[i] and 
                close[i] > ema50_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price < ETF (bearish) + price < 1d EMA50 + volume confirmation
            elif (close[i] < etf[i] and 
                  close[i] < ema50_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below ETF
            if close[i] < etf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above ETF
            if close[i] > etf[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals