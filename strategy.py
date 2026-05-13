#!/usr/bin/env python3
# 6h_KAMA_RSI_Trend_Filtered_Breakout
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 6h as trend filter with RSI(2) for mean-reversion entries.
# Go long when price is above KAMA and RSI(2) < 10 (oversold bounce in uptrend).
# Go short when price is below KAMA and RSI(2) > 90 (overbought pullback in downtrend).
# Uses 1d trend (EMA50) as higher timeframe filter to ensure alignment with daily momentum.
# Volume confirmation reduces false signals. Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
# Target: 20-40 trades/year per symbol to minimize fee drag.

name = "6h_KAMA_RSI_Trend_Filtered_Breakout"
timeframe = "6h"
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

    # 6h KAMA (trend filter)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # 6h RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA (uptrend) + RSI(2) < 10 + volume confirm + 1d uptrend
            if (close[i] > kama[i] and 
                rsi[i] < 10 and 
                volume_confirm[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA (downtrend) + RSI(2) > 90 + volume confirm + 1d downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] > 90 and 
                  volume_confirm[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI(2) > 50 (overbought) or price < KAMA (trend change)
            if rsi[i] > 50 or close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI(2) < 50 (oversold) or price > KAMA (trend change)
            if rsi[i] < 50 or close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals