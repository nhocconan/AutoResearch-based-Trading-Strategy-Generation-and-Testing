#!/usr/bin/env python3
"""
4h_KAMA_Direction_Trend_Filter_1dTrend_Volume
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a robust trend filter. Combined with 1d EMA50 trend and volume confirmation, it captures trend continuation in both bull and bear markets while avoiding whipsaws in chop. The adaptive nature reduces false signals during high volatility, improving win rate.
"""

name = "4h_KAMA_Direction_Trend_Filter_1dTrend_Volume"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average) on 4h close
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 1))
    change[0] = 0  # first period has no change
    direction = np.abs(close - np.roll(close, 10))
    volatility = pd.Series(change).rolling(window=10, min_periods=10).sum().values
    er = np.where(volatility != 0, direction / volatility, 0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # start after ER period
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: volume > 1.3x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        if np.isnan(kama[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + 1d uptrend + volume spike
            if close[i] > kama[i] and close[i] > ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + 1d downtrend + volume spike
            elif close[i] < kama[i] and close[i] < ema50_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.3:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below KAMA or 1d trend turns down
            if close[i] < kama[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above KAMA or 1d trend turns up
            if close[i] > kama[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals