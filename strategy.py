#!/usr/bin/env python3
# 4h_KAMA_Trend_Plus_Volume_Spike
# Hypothesis: KAMA trend direction combined with volume spikes captures momentum with controlled trade frequency.
# KAMA adapts to market noise, reducing false signals in choppy conditions.
# Works in bull markets via up-trend entries and bear markets via down-trend entries.
# Volume spike confirms institutional participation, reducing false breakouts.
# Target: 20-50 trades per year per symbol to minimize fee drag.

name = "4h_KAMA_Trend_Plus_Volume_Spike"
timeframe = "4h"
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

    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)

    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_length))
    change[0:er_length] = 0
    gap = np.abs(np.diff(close, prepend=close[0]))
    er = change / np.where(gap.rolling(window=er_length, min_periods=1).sum() == 0, 1, gap.rolling(window=er_length, min_periods=1).sum())
    er = np.where(np.isnan(er), 0, er)

    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc = np.where(np.isnan(sc), 0, sc)

    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])

    # Volume filter: >2.0x 20-period average
    vol_avg_20 = np.zeros_like(volume)
    vol_sum = np.cumsum(volume)
    vol_sum[0:20] = 0
    vol_avg_20[20:] = (vol_sum[20:] - vol_sum[0:-20]) / 20
    vol_avg_20[0:20] = np.nan

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(kama[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above KAMA + volume spike
            if close[i] > kama[i] and volume[i] > vol_avg_20[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + volume spike
            elif close[i] < kama[i] and volume[i] > vol_avg_20[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below KAMA or volume drop
            if close[i] < kama[i] or volume[i] < vol_avg_20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA or volume drop
            if close[i] > kama[i] or volume[i] < vol_avg_20[i] * 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals