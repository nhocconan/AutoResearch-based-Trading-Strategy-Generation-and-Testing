#!/usr/bin/env python3
# 1d_1W_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: Daily KAMA trend filter with weekly trend confirmation and volume spike.
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation ensures institutional participation.
# Designed for 10-30 trades/year to minimize fee drag.

name = "1d_1W_KAMA_Trend_With_Volume_Confirmation"
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
    if len(df_1w) < 50:
        return np.zeros(n)

    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0) if False else None
    # Proper ER calculation: |current - close 10 periods ago| / sum of absolute changes over 10 periods
    lookback = 10
    abs_change = np.abs(np.diff(close_1w, lookback))  # |close[t] - close[t-lookback]|
    abs_change = np.concatenate([np.full(lookback, np.nan), abs_change])
    abs_diff = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    sum_abs_diff = np.convolve(abs_diff, np.ones(lookback), 'same')  # sum of abs changes over lookback
    er = np.where(sum_abs_diff > 0, abs_change / sum_abs_diff, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[lookback] = close_1w[lookback]  # seed
    for i in range(lookback + 1, len(close_1w)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)

    # Calculate daily KAMA for entry signal
    abs_change_d = np.abs(np.diff(close, lookback))
    abs_change_d = np.concatenate([np.full(lookback, np.nan), abs_change_d])
    abs_diff_d = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_diff_d = np.convolve(abs_diff_d, np.ones(lookback), 'same')
    er_d = np.where(sum_abs_diff_d > 0, abs_change_d / sum_abs_diff_d, 0)
    sc_d = (er_d * (fast_sc - slow_sc) + slow_sc) ** 2
    kama_d = np.full_like(close, np.nan)
    kama_d[lookback] = close[lookback]
    for i in range(lookback + 1, len(close)):
        if not np.isnan(sc_d[i]) and not np.isnan(kama_d[i-1]):
            kama_d[i] = kama_d[i-1] + sc_d[i] * (close[i] - kama_d[i-1])
        else:
            kama_d[i] = kama_d[i-1]

    # Volume confirmation: current volume > 2.0x average of last 20 periods
    vol_ma = np.convolve(volume, np.ones(20)/20, 'same')
    vol_ma[:10] = np.nan
    vol_ma[-9:] = np.nan
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(kama_d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Daily trend: price vs daily KAMA
        bullish = close[i] > kama_d[i]
        bearish = close[i] < kama_d[i]
        # Weekly trend: price vs weekly KAMA
        weekly_bullish = close[i] > kama_1w_aligned[i]
        weekly_bearish = close[i] < kama_1w_aligned[i]

        if position == 0:
            # LONG: Price above daily KAMA, weekly trend bullish, volume confirmation
            if bullish and weekly_bullish and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below daily KAMA, weekly trend bearish, volume confirmation
            elif bearish and weekly_bearish and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below daily KAMA or weekly trend turns bearish
            if not bullish or not weekly_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above daily KAMA or weekly trend turns bullish
            if not bearish or not weekly_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals