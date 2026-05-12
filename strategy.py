#!/usr/bin/env python3
# 1d_KAMA_Trend_With_Volume_Confirmation
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) trend direction with volume spike confirmation on daily timeframe.
# Long when KAMA slope is positive and volume > 1.5x 20-day average volume.
# Short when KAMA slope is negative and volume > 1.5x 20-day average volume.
# Exit when KAMA slope changes sign or volume drops below threshold.
# Designed for low trade frequency (7-25/year) to avoid fee drag. Works in both bull and bear markets by following adaptive trend.

name = "1d_KAMA_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter (optional, can be removed if not needed)
    # For now, we focus on daily KAMA and volume

    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Parameters: ER fast/slow, lookback for volatility
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    window = 10             # ER lookback period

    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=window))  # |close[t] - close[t-window]|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of |close[t] - close[t-1]| over window
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=0)

    # Volume spike threshold: 1.5x 20-day average volume
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # start after volume SMA warmup
        # Skip if any required data is NaN
        if np.isnan(kama_slope[i]) or np.isnan(volume_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: positive KAMA slope and volume spike
            if kama_slope[i] > 0 and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: negative KAMA slope and volume spike
            elif kama_slope[i] < 0 and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA slope turns negative or volume drops
            if kama_slope[i] <= 0 or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA slope turns positive or volume drops
            if kama_slope[i] >= 0 or volume[i] <= volume_threshold[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals