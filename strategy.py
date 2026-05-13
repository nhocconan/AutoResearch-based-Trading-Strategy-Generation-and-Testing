#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_Spike
Hypothesis: KAMA adapts to market noise, providing a dynamic trend filter that works in both bull and bear markets.
Price crossing above/below KAMA with volume confirmation captures trend changes. Uses 1d trend filter for higher timeframe alignment.
Designed for low-frequency, high-quality setups to minimize fee drag on 12h timeframe.
Target: 12-37 trades/year (50-150 over 4 years).
"""

name = "12h_KAMA_Trend_With_Volume_Spike"
timeframe = "12h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Handle first 9 values where diff(10) is not available
    er = np.zeros_like(close)
    er[9:] = change[9:] / np.maximum(volatility[9:], 1e-10)
    # Smoothing constants
    fastest = 2 / (2 + 1)   # for fast EMA (2)
    slowest = 2 / (30 + 1)  # for slow EMA (30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume spike: volume > 2.0 * 20-period average (~10 days at 12h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA and above 1d EMA50 (uptrend) with volume spike
            if close[i] > kama[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA and below 1d EMA50 (downtrend) with volume spike
            elif close[i] < kama[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA or trend turns bearish
            if close[i] < kama[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA or trend turns bullish
            if close[i] > kama[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals