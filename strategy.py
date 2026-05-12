#!/usr/bin/env python3
"""
12h_RSIBounce_1wTrend
Hypothesis: In a strong weekly trend (price > weekly EMA200), buy RSI(12h) pullbacks below 30 and sell when RSI > 70 or trend weakens. Inverse for short: sell when weekly trend is down (price < weekly EMA200) and RSI > 70, buy back when RSI < 30. Volume confirmation filters weak signals. Designed to capture mean reversion within strong trends, working in both bull and bear markets via weekly trend filter.
"""

name = "12h_RSIBounce_1wTrend"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # RSI(14) on 12h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly uptrend (price > weekly EMA200) + RSI oversold (<30) + volume spike
            if close[i] > ema200_1w_aligned[i] and rsi[i] < 30 and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend (price < weekly EMA200) + RSI overbought (>70) + volume spike
            elif close[i] < ema200_1w_aligned[i] and rsi[i] > 70 and volume[i] > vol_avg_20[i] * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or weekly trend turns down (price < weekly EMA200)
            if rsi[i] > 70 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or weekly trend turns up (price > weekly EMA200)
            if rsi[i] < 30 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals