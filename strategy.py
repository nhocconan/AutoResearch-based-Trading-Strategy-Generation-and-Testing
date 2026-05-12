#!/usr/bin/env python3
# 1d_1W_KAMA_Trend_Volume_Confirmation
# Hypothesis: Daily trend following using Kaufman Adaptive Moving Average (KAMA) with weekly trend filter and volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in choppy markets while capturing trends.
# Weekly trend filter ensures alignment with higher-timeframe momentum, avoiding counter-trend trades.
# Volume confirmation ensures institutional participation, filtering false breakouts.
# Designed for low trade frequency (10-25 trades/year) to minimize fee drag and improve generalization.

name = "1d_1W_KAMA_Trend_Volume_Confirmation"
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
    # Calculate Efficiency Ratio (ER) and Smoothing Constants
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w, prepend=close_1w[0])), axis=0)  # Placeholder for correct calculation
    # Correct volatility calculation: sum of absolute changes over lookback period
    volatility = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1w[max(0, i-9):i+1])))  # 10-period ER
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6665 - 0.0645) + 0.0645) ** 2  # Smoothing constant
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)

    # Volume confirmation: current volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Weekly trend filter based on KAMA
        bullish_trend = close[i] > kama_1w_aligned[i]
        bearish_trend = close[i] < kama_1w_aligned[i]

        if position == 0:
            # LONG: Price above weekly KAMA with volume confirmation
            if bullish_trend and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below weekly KAMA with volume confirmation
            elif bearish_trend and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly KAMA
            if not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly KAMA
            if not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals