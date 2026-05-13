#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_MeanReversion
# Hypothesis: KAMA identifies trend direction on 12h, RSI(14) identifies overbought/oversold conditions for mean reversion entries.
# Works in bull (trend-following: long when KAMA up, RSI < 30) and bear (counter-trend: short when KAMA down, RSI > 70).
# Volume confirmation filters false signals. Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_KAMA_Trend_RSI_MeanReversion"
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

    # Get 1d data for KAMA trend
    df_1d = get_htf_data(prices, '1d')
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    change = abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close']))
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'][0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'][i] - kama[i-1])
    kama = kama.astype(np.float64)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)

    # Get 1d data for RSI
    delta = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)

    # Volume filter: >1.3x 20-period average
    vol_avg_20 = pd.Series(volume).ewm(span=20, adjust=False).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA trending up + RSI oversold + volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and
                volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA trending down + RSI overbought + volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and
                  volume[i] > vol_avg_20[i] * 1.3):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI overbought
            if (close[i] < kama_aligned[i] or rsi_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI oversold
            if (close[i] > kama_aligned[i] or rsi_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals