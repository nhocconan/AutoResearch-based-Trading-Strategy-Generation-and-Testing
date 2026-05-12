#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: Use Kaufman Adaptive Moving Average (KAMA) for trend direction on 12h timeframe.
# Enter long when price > KAMA and RSI > 50 in low-chop regime; short when price < KAMA and RSI < 50 in low-chop regime.
# Use 1-day Chop Index (ER < 38.2) as regime filter to avoid whipsaws in ranging markets.
# Exit when trend reverses or chop increases. Designed for fewer trades (target: 20-40/year) with trend-following edge in both bull and bear markets.

name = "12h_KAMA_Trend_RSI_Chop_Filter"
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

    # Get daily data for Chop Index (ER-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 12-period ER (Efficiency Ratio) for Chop Index
    change = np.abs(df_1d['close'].diff(12).values)
    volatility = np.abs(df_1d['close'].diff(1)).values
    vol_sum = pd.Series(volatility).rolling(window=12, min_periods=12).sum().values
    er = np.divide(change, vol_sum, out=np.zeros_like(change), where=vol_sum!=0)
    # Smooth ER with smoothing constants (fastest EMA=2, slowest=30)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1d = np.full_like(df_1d['close'], np.nan)
    kama_1d[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        if np.isnan(sc[i]):
            kama_1d[i] = kama_1d[i-1]
        else:
            kama_1d[i] = kama_1d[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama_1d[i-1])
    chop_1d = 100 * (1 - er)  # Chop Index: higher = more choppy
    chop_filter = chop_1d < 38.2  # Low chop = trending regime

    # Align chop filter to 12h timeframe
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))

    # Calculate KAMA on 12h price data (fast=2, slow=30)
    change_12h = np.abs(np.diff(close, prepend=close[0]))
    volatility_12h = np.abs(np.diff(close, prepend=close[0]))
    vol_sum_12h = pd.Series(volatility_12h).rolling(window=12, min_periods=12).sum().values
    er_12h = np.divide(change_12h, vol_sum_12h, out=np.zeros_like(change_12h), where=vol_sum_12h!=0)
    sc_12h = (er_12h * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_12h = np.full_like(close, np.nan)
    kama_12h[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc_12h[i]):
            kama_12h[i] = kama_12h[i-1]
        else:
            kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close[i] - kama_12h[i-1])

    # RSI (14) on 12h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_12h[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade in low-chop (trending) regime
        if chop_filter_aligned[i] < 0.5:  # False = high chop
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price > KAMA and RSI > 50
            if close[i] > kama_12h[i] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # SHORT: price < KAMA and RSI < 50
            elif close[i] < kama_12h[i] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < KAMA or RSI < 50
            if close[i] < kama_12h[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > KAMA or RSI > 50
            if close[i] > kama_12h[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals