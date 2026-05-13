#!/usr/bin/env python3
# 12h_KAMA_Direction_1dTrend_VolumeSpike
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a smooth trend line.
# In trending markets, price stays close to KAMA; in ranging markets, it diverges.
# We go long when price crosses above KAMA with volume spike and 1d uptrend (price > 1d EMA34).
# We go short when price crosses below KAMA with volume spike and 1d downtrend (price < 1d EMA34).
# Exit when price crosses back over KAMA.
# Uses 12h timeframe with 1d trend filter to reduce whipsaw and capture major trends.
# Designed to work in bull (buy in uptrend) and bear (sell in downtrend) markets.
# Target: 15-30 trades/year per symbol.

name = "12h_KAMA_Direction_1dTrend_VolumeSpike"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average) on 12h close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[i] - close[i-1]| over 10 periods
    # Avoid division by zero
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]

    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after first ER calculation

    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # Get 1d EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if data is not ready
        if (np.isnan(kama[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price crosses above KAMA with volume spike and 1d EMA uptrend
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price crosses below KAMA with volume spike and 1d EMA downtrend
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA
            if close[i] < kama[i] and close[i-1] >= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA
            if close[i] > kama[i] and close[i-1] <= kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals