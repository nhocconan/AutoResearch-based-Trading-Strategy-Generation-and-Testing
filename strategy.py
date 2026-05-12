#!/usr/bin/env python3
"""
4h_RSI_14_EMA50_Filter_With_Volume
Hypothesis: RSI(14) above 50 with EMA(50) filter on 4h timeframe, combined with volume confirmation (1.3x average) and 1d trend filter, captures momentum while avoiding whipsaws. Works in bull markets (trend following) and bear markets (mean reversion via RSI extremes). Uses 4h as primary timeframe with 1d for trend filter.
"""

name = "4h_RSI_14_EMA50_Filter_With_Volume"
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

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # RSI(14) on 4h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # EMA(50) on 4h
    ema_50_4h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume spike: >1.3x 20-period average (4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 50, price > EMA50(4h), 1d uptrend, volume spike
            if (rsi[i] > 50 and 
                close[i] > ema_50_4h[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 50, price < EMA50(4h), 1d downtrend, volume spike
            elif (rsi[i] < 50 and 
                  close[i] < ema_50_4h[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 40 or price < EMA50(4h)
            if rsi[i] < 40 or close[i] < ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 60 or price > EMA50(4h)
            if rsi[i] > 60 or close[i] > ema_50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals