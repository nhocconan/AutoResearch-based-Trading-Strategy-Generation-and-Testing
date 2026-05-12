#!/usr/bin/env python3
# 4h_KAMA_1dTrend_Volume_Stop
# Hypothesis: KAMA trend on 4h with 1d trend filter and volume confirmation.
# Uses KAMA for adaptive trend following on 4h, confirmed by 1d EMA trend and volume spike.
# Stops when trend reverses. Designed for 20-50 trades/year with low turnover.

name = "4h_KAMA_1dTrend_Volume_Stop"
timeframe = "4h"
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
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Calculate KAMA on 4h
    # Efficiency ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Align 1d EMA to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters
        bullish = kama[i] > close[i] and ema_1d_aligned[i] < close[i]
        bearish = kama[i] < close[i] and ema_1d_aligned[i] > close[i]

        if position == 0:
            # LONG: KAMA below price and 1d trend up with volume
            if bullish and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA above price and 1d trend down with volume
            elif bearish and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA crosses above price or 1d trend turns down
            if not bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA crosses below price or 1d trend turns up
            if not bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals