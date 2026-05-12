#!/usr/bin/env python3
# 12h_1D_KAMA_Trend_Filter_1dTrend_Volume
# Hypothesis: Use 12h KAMA to capture the dominant trend, with daily trend confirmation and volume filter.
# KAMA adapts to market conditions - faster in trends, slower in ranges, reducing whipsaws.
# Daily trend filter ensures alignment with higher timeframe momentum.
# Volume confirmation ensures breakouts have institutional participation.
# Designed for 12h timeframe to limit trade frequency (target: 15-30 trades/year).

name = "12h_1D_KAMA_Trend_Filter_1dTrend_Volume"
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

    # Get 12h data for KAMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)

    # 12h KAMA (adaptive moving average)
    close_12h = df_12h['close']
    # Calculate efficiency ratio
    change = abs(close_12h - close_12h.shift(10))
    volatility = abs(close_12h - close_12h.shift(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan, dtype=float)
    kama[0] = close_12h.iloc[0]
    for i in range(1, len(close_12h)):
        if not np.isnan(sc.iloc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc.iloc[i] * (close_12h.iloc[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume confirmation: current volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Trend filters
        kama_trend = close[i] > kama_12h_aligned[i]
        daily_trend = close[i] > ema_34_1d_aligned[i]

        if position == 0:
            # LONG: Price above both KAMA and daily EMA with volume confirmation
            if kama_trend and daily_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below both KAMA and daily EMA with volume confirmation
            elif not kama_trend and not daily_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or daily trend fails
            if not kama_trend or not daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or daily trend fails
            if kama_trend or daily_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals