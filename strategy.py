#!/usr/bin/env python3
# 1d_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a robust trend filter. Combined with RSI(14) for momentum confirmation on the daily timeframe, this strategy captures sustained trends while avoiding whipsaws in choppy markets. The weekly trend (from 1w data) acts as a regime filter to ensure alignment with the higher-timeframe direction. Designed for low trade frequency (~10-25 trades/year) to minimize fee drag, suitable for both bull and bear markets by following the dominant trend.

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # KAMA on weekly close
    kama_1w = calculate_kama(close_1w, 30, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    # Daily indicators
    # Daily KAMA for entry
    kama = calculate_kama(close, 30, 2, 30)
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if weekly KAMA or daily RSI not ready
        if np.isnan(kama_1w_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above weekly KAMA (bullish regime) and RSI > 50 (bullish momentum)
            if close[i] > kama_1w_aligned[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # SHORT: price below weekly KAMA (bearish regime) and RSI < 50 (bearish momentum)
            elif close[i] < kama_1w_aligned[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below weekly KAMA or RSI < 40 (loss of momentum)
            if close[i] < kama_1w_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above weekly KAMA or RSI > 60 (loss of momentum)
            if close[i] > kama_1w_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

def calculate_kama(close, er_period, fast_sc, slow_sc):
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    # KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

from mtf_data import get_htf_data, align_htf_to_ltf