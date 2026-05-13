#!/usr/bin/env python3
# 1d_RSI2_Close_Bounce_1wTrend
# Mean reversion on daily timeframe: enter long when RSI(2) < 10 and price closes below 20-day low,
# but only in weekly uptrend (price > weekly EMA40). Enter short when RSI(2) > 90 and price closes
# above 20-day high in weekly downtrend. Exit when RSI(2) crosses 50.
# Designed for low trade frequency and high win rate in both bull and bear markets.

name = "1d_RSI2_Close_Bounce_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Weekly trend filter: EMA40 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)

    # Daily indicators
    rsi_period = 2
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # 20-day high/low for entry filter
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        if np.isnan(rsi[i]) or np.isnan(ema40_1w_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI(2) < 10 + close below 20-day low + weekly uptrend
            if (rsi[i] < 10 and 
                close[i] < low_20[i] and 
                close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI(2) > 90 + close above 20-day high + weekly downtrend
            elif (rsi[i] > 90 and 
                  close[i] > high_20[i] and 
                  close[i] < ema40_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals