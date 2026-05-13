#!/usr/bin/env python3
# 1d_KAMA_Trend_With_RSI_Filter_1wTrend
# Hypothesis: KAMA trend on 1d for direction, RSI(14) for momentum confirmation, and 1-week trend filter to ensure alignment with higher timeframe.
# KAMA adapts to market noise, reducing false signals in choppy markets. RSI filters overextended entries.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, improving win rate in both bull and bear markets.
# Designed for 10-25 trades/year to minimize fee drag on 1d timeframe.

name = "1d_KAMA_Trend_With_RSI_Filter_1wTrend"
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

    # Calculate 1-week EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Calculate KAMA on 1d data (ER=10, fast=2, slow=30)
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constant
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_aligned = kama  # Already on 1d timeframe

    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend first value to match length
    rsi = np.concatenate([[50], rsi])

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA, RSI > 50 (but not overbought), and above weekly EMA50
            if (close[i] > kama[i] and 
                50 < rsi[i] < 70 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA, RSI < 50 (but not oversold), and below weekly EMA50
            elif (close[i] < kama[i] and 
                  30 < rsi[i] < 50 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or RSI > 70 (overbought) or below weekly EMA50
            if (close[i] < kama[i] or 
                rsi[i] >= 70 or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or RSI < 30 (oversold) or above weekly EMA50
            if (close[i] > kama[i] or 
                rsi[i] <= 30 or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals