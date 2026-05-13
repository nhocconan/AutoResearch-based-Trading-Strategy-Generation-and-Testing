#!/usr/bin/env python3
# 1d_KAMA_Trend_With_RSI_Confirmation
# Hypothesis: Use Kaufman's Adaptive Moving Average (KAMA) to capture medium-term trends
# on the daily timeframe, with RSI filtering to avoid overextended entries.
# Long when KAMA turns up and RSI < 70 (not overbought); short when KAMA turns down
# and RSI > 30 (not oversold). Exit on opposite KAMA turn.
# Designed for low trade frequency (<25/year) to minimize fee drag, works in bull and bear
# markets by following adaptive trend with momentum filter.

name = "1d_KAMA_Trend_With_RSI_Confirmation"
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
    volume = prices['volume'].values

    # --- KAMA (Kaufman Adaptive Moving Average) ---
    # ER = Efficiency Ratio, SC = Smoothing Constant
    # Using 10-day ER as in standard KAMA
    change = np.abs(np.diff(close, k=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of abs changes over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # --- RSI (14-period) ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # First 14 values are NaN due to min_periods

    # --- Signals ---
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(15, n):  # start after RSI warmup
        if np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up (current > previous) AND RSI not overbought
            if kama[i] > kama[i-1] and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down (current < previous) AND RSI not oversold
            elif kama[i] < kama[i-1] and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals