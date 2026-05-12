#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter
# Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
# combined with RSI for momentum confirmation and Choppiness Index for regime filtering.
# Enter long when KAMA upward, RSI > 50, and market is trending (CHOP < 38.2).
# Enter short when KAMA downward, RSI < 50, and market is trending (CHOP < 38.2).
# Exit when trend reverses or market becomes choppy (CHOP >= 38.2).
# Designed for low trade frequency (target: 10-25 trades/year) to minimize fee drag
# and work in both bull and bear markets via adaptive trend and regime filters.

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Calculate Efficiency Ratio (ER) for KAMA over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0  # first 10 values invalid
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly: sum of absolute daily changes over 10 periods
    volatility = np.zeros_like(close)
    for i in range(10, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[9] = close[9]  # start at index 9
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Calculate RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # align length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Calculate Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first bar
    # Sum of true ranges over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.where(atr_sum != 0, -100 * np.log10((hh - ll) / atr_sum) / np.log10(14), 50)
    # Handle edge cases where hh == ll
    chop = np.where((hh - ll) == 0, 50, chop)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA upward (close > kama), RSI > 50, trending market (CHOP < 38.2)
            if (close[i] > kama[i] and 
                rsi[i] > 50 and
                chop[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA downward (close < kama), RSI < 50, trending market (CHOP < 38.2)
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and
                  chop[i] < 38.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns downward OR market becomes choppy
            if (close[i] < kama[i]) or (chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns upward OR market becomes choppy
            if (close[i] > kama[i]) or (chop[i] >= 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals