#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Confirmation
Hypothesis: On 1d timeframe, buy when KAMA trend turns up and volume > 2x average, sell when KAMA trend turns down and volume > 2x average. Uses volume confirmation to avoid false trend changes, targeting low trade frequency (<25/year) to minimize fee drag while capturing trends in both bull and bear markets.
"""

name = "1d_KAMA_Trend_With_Volume_Confirmation"
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

    # Get weekly data for trend filter (optional, but can add confluence)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # KAMA parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle edge cases for ER calculation
    er = np.zeros_like(close)
    for i in range(er_len, len(close)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0

    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume confirmation: volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Weekly EMA for trend filter (optional)
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or np.isnan(kama[i-1]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA turning up + volume spike
            if (kama[i] > kama[i-1] and 
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA turning down + volume spike
            elif (kama[i] < kama[i-1] and 
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down
            if kama[i] < kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up
            if kama[i] > kama[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals