#!/usr/bin/env python3
# 4h_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# RSI filter avoids entries during extreme overbought/oversold conditions, reducing whipsaw.
# Uses 1-day EMA34 as higher timeframe trend filter to avoid counter-trend trades.
# Target: 20-40 trades/year.

name = "4h_KAMA_Trend_With_RSI_Filter"
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

    # KAMA (Kaufman Adaptive Moving Average)
    def kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[:period] = close[:period].mean()
        for i in range(period, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama

    # RSI
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
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Indicators
    kama_val = kama(close, 10, 2, 30)
    rsi_val = rsi(close, 14)

    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA + RSI < 70 (not overbought) + volume spike + 1d uptrend
            if (close[i] > kama_val[i] and 
                rsi_val[i] < 70 and
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA + RSI > 30 (not oversold) + volume spike + 1d downtrend
            elif (close[i] < kama_val[i] and 
                  rsi_val[i] > 30 and
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA or RSI > 80 (overbought exit)
            if (close[i] < kama_val[i] or rsi_val[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA or RSI < 20 (oversold exit)
            if (close[i] > kama_val[i] or rsi_val[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals