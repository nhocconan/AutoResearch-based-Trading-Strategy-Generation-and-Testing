#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_OverboughtOversold
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction. Combined with RSI extremes for mean-reversion entries in the direction of the trend, this strategy captures swing moves while avoiding whipsaws. Works in both bull and bear markets by using trend filter and avoiding counter-trend trades.

name = "4h_KAMA_Trend_RSI_OverboughtOversold"
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

    # Calculate KAMA for trend (using close)
    def calculate_kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change) > 1 else np.abs(change)
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama = calculate_kama(close, 10, 2, 30)

    # RSI for mean reversion signals
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    rsi = calculate_rsi(close, 14)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA (uptrend) + RSI oversold (<30) + volume confirmation
            if (close[i] > kama[i] and 
                rsi[i] < 30 and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA (downtrend) + RSI overbought (>70) + volume confirmation
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or RSI overbought (>70) or volume drop
            if (close[i] < kama[i] or 
                rsi[i] > 70 or
                volume[i] < vol_avg_20[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or RSI oversold (<30) or volume drop
            if (close[i] > kama[i] or 
                rsi[i] < 30 or
                volume[i] < vol_avg_20[i] * 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals