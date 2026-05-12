#/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) on 12h to detect trend direction, confirmed by 14-period RSI on 1d (overbought/oversold filter) and volume spikes (>2x 20-period average). Enter long when price > KAMA and RSI < 70 with volume spike; short when price < KAMA and RSI > 30 with volume spike. Exit on KAMA crossover reverse. Targets 12-37 trades/year to minimize fee decay and work in both bull/bear markets via trend filter and RSI filter.

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
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

    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate KAMA on 12h
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, 10))  # 10-period change
    change[0:10] = 0  # first 10 periods undefined
    volatility = np.abs(np.diff(close, prepend=close[0]))  # daily volatility
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = np.where(vol_sum != 0, change / vol_sum, 0)

    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30

    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # RSI on 1d (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)

    # Volume confirmation: volume > 2x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # start after KAMA warmup
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > KAMA + RSI < 70 (not overbought) + volume spike
            if (close[i] > kama[i] and 
                rsi_1d_aligned[i] < 70 and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA + RSI > 30 (not oversold) + volume spike
            elif (close[i] < kama[i] and 
                  rsi_1d_aligned[i] > 30 and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals