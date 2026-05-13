#!/usr/bin/env python3
# 1d_KAMA_Trend_RSI_1dTrend
# Hypothesis: Use 1d KAMA for trend direction, RSI for momentum confirmation, and price action for entry.
# Long when price crosses above KAMA in uptrend with RSI > 50, short when price crosses below KAMA in downtrend with RSI < 50.
# Exit when price crosses back below/above KAMA.
# Designed for low trade frequency (<100 total trades over 4 years) with clear trend-following logic.

name = "1d_KAMA_Trend_RSI_1dTrend"
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
    volume = prices['volume'].values

    # Get 1d data (same as primary timeframe) for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close'])).rolling(window=10, min_periods=10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Align KAMA to 1d timeframe (no alignment needed as it's same timeframe)
    kama_aligned = kama

    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above KAMA + RSI > 50 (bullish momentum)
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below KAMA + RSI < 50 (bearish momentum)
            elif close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA
            if close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA
            if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals