#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_Filter
# Hypothesis: On 12h timeframe, enter long when KAMA turns bullish (price > KAMA) and RSI(14) > 50, enter short when KAMA turns bearish (price < KAMA) and RSI(14) < 50. Uses 1d EMA50 as trend filter to avoid counter-trend trades. Designed for low trade frequency (15-25/year) to minimize fee drag and improve robustness in both bull and bear markets. Uses discrete position sizing (0.25).

name = "12h_KAMA_Direction_RSI_Filter"
timeframe = "12h"
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

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    def kama(close, length=10, fast=2, slow=30):
        # Calculate Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close))
        er = np.zeros_like(close)
        for i in range(len(close)):
            if i == 0:
                er[i] = 0
            else:
                er[i] = change[i] / (np.sum(volatility[max(0, i-length+1):i+1]) + 1e-10)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # Calculate KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals

    kama_vals = kama(close, length=10, fast=2, slow=30)

    # Calculate RSI(14)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length-1] = np.mean(gain[:length])
        avg_loss[length-1] = np.mean(loss[:length])
        for i in range(length, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals

    rsi_vals = rsi(close, length=14)

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA (bullish) AND RSI > 50 AND price > 1d EMA50 (uptrend)
            if (close[i] > kama_vals[i] and 
                rsi_vals[i] > 50 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA (bearish) AND RSI < 50 AND price < 1d EMA50 (downtrend)
            elif (close[i] < kama_vals[i] and 
                  rsi_vals[i] < 50 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA OR RSI < 50 OR trend turns down
            if (close[i] < kama_vals[i] or 
                rsi_vals[i] < 50 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA OR RSI > 50 OR trend turns up
            if (close[i] > kama_vals[i] or 
                rsi_vals[i] > 50 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals