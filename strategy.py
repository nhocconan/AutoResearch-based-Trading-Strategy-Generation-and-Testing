#!/usr/bin/env python3
"""
4h_KAMA_Trend_RSI_MeanReversion
Hypothesis: On 4h timeframe, KAMA determines trend direction (bullish if price > KAMA, bearish if price < KAMA). In bullish trend, look for mean reversion entries when RSI < 30; in bearish trend, look for RSI > 70. Uses volume confirmation (volume > 1.5x 20-period average) to filter low-quality signals. Designed to work in both bull and bear markets by adapting to trend direction and avoiding trades in sideways markets via RSI extremes. Targets 20-40 trades per year to minimize fee drag.
"""

name = "4h_KAMA_Trend_RSI_MeanReversion"
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

    # Calculate KAMA (Kaufman Adaptive Moving Average) for trend
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    kama_vals = kama(close, 10, 2, 30)

    # Calculate RSI (Relative Strength Index) for mean reversion
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals

    rsi_vals = rsi(close, 14)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish trend (price > KAMA) + RSI oversold (<30) + volume confirmation
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] < 30 and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish trend (price < KAMA) + RSI overbought (>70) + volume confirmation
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] > 70 and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend turns bearish (price < KAMA)
            if rsi_vals[i] > 70 or close[i] < kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend turns bullish (price > KAMA)
            if rsi_vals[i] < 30 or close[i] > kama_vals[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals