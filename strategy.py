#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_And_Chop
Hypothesis: Use daily KAMA for trend direction, RSI for momentum filter, and Choppiness Index for regime filter.
Enter long when KAMA is rising, RSI > 50, and market is trending (CHOP < 38.2).
Enter short when KAMA is falling, RSI < 50, and market is trending (CHOP < 38.2).
Exit when trend changes or RSI reverts to neutral.
Designed for low frequency (target: 15-25 trades/year) to avoid fee drag, works in both bull and bear markets via regime filter.
"""

name = "1d_KAMA_Trend_With_RSI_And_Chop"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate KAMA (Kaufman Adaptive Moving Average)
    close_series = pd.Series(close)
    # Efficiency Ratio (ER)
    change = abs(close - close.shift(10))
    volatility = abs(close.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2
    # Handle NaN/inf
    sc = sc.fillna(0)
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])

    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined

    # Calculate Choppiness Index (CHOP)
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.sum(atr) / (max_high - min_low)) / np.log10(14)
    # Handle division by zero or invalid
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # default to neutral

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising, RSI > 50, trending market (CHOP < 38.2)
            if (kama[i] > kama[i-1] and
                rsi[i] > 50 and
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, trending market (CHOP < 38.2)
            elif (kama[i] < kama[i-1] and
                  rsi[i] < 50 and
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI <= 50
            if (kama[i] < kama[i-1]) or (rsi[i] <= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI >= 50
            if (kama[i] > kama[i-1]) or (rsi[i] >= 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals