#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Trend_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear regimes.
Combined with RSI(14) > 50 for long and < 50 for short filters, we avoid counter-trend trades.
This reduces whipsaws and focuses on momentum-aligned entries, suitable for 1d timeframe with low trade frequency.
"""

name = "1d_KAMA_Direction_RSI_Trend_Filter"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Calculate KAMA on daily close
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum()
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama = calculate_kama(close, 10, 2, 30)

    # Calculate RSI(14)
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(close, 14)

    # Weekly Supertrend-like filter using ATR
    def calculate_atr(high, low, close, length=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean().values
        return atr

    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    # Simple trend: price above/below average of high/low
    avg_price_1w = (high_1w + low_1w) / 2
    trend_up = close_1w > avg_price_1w
    trend_down = close_1w < avg_price_1w

    # Align weekly trend to daily
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, trend_down.astype(float))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # warmup for KAMA/RSI
        kama_val = kama[i]
        rsi_val = rsi[i]
        trend_up_val = trend_up_aligned[i]
        trend_down_val = trend_down_aligned[i]

        if np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(trend_up_val) or np.isnan(trend_down_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + RSI > 50 + weekly uptrend
            if close[i] > kama_val and rsi_val > 50 and trend_up_val > 0.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + RSI < 50 + weekly downtrend
            elif close[i] < kama_val and rsi_val < 50 and trend_down_val > 0.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or RSI < 40
            if close[i] < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or RSI > 60
            if close[i] > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals