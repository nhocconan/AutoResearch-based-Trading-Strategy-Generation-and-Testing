#!/usr/bin/env python3
# 1d_HMA_Trend_Breakout_1wTrend_Filter_Volume
# Hypothesis: Price breaking above/below 1d HMA(16) with 1w HMA(16) trend filter and volume confirmation
# captures momentum while minimizing trades. Works in bull via breakouts above HMA and in bear via breakdowns.
# Uses 1w HMA to filter long-term trend and volume spike for confirmation, reducing false signals.
# Target: 10-25 trades per year per symbol to minimize fee drag.

name = "1d_HMA_Trend_Breakout_1wTrend_Filter_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hma(arr, period):
    """Hull Moving Average"""
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = np.zeros_like(arr)
    wma1 = np.zeros_like(arr)
    wma = np.zeros_like(arr)
    for i in range(len(arr)):
        if i < period:
            wma[i] = np.nan
            continue
        # WMA of half period
        sum_w = 0.0
        sum_v = 0.0
        for j in range(half):
            weight = half - j
            sum_w += weight
            sum_v += weight * arr[i - j]
        wma2[i] = sum_v / sum_w if sum_w != 0 else np.nan
        # WMA of full period
        sum_w = 0.0
        sum_v = 0.0
        for j in range(period):
            weight = period - j
            sum_w += weight
            sum_v += weight * arr[i - j]
        wma1[i] = sum_v / sum_w if sum_w != 0 else np.nan
        # Hull MA
        raw = 2 * wma2[i] - wma1[i]
        if i < sqrt:
            wma[i] = np.nan
            continue
        sum_w = 0.0
        sum_v = 0.0
        for j in range(sqrt):
            weight = sqrt - j
            sum_w += weight
            sum_v += weight * raw
        wma[i] = sum_v / sum_w if sum_w != 0 else np.nan
    return wma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # HMA(16) for 1d
    hma_1d = hma(close, 16)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    hma_1w = hma(df_1w['close'].values, 16)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(hma_1d[i]) or np.isnan(hma_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above 1d HMA + 1w HMA uptrend + volume spike
            if (close[i] > hma_1d[i] and 
                hma_1w_aligned[i] > hma_1w_aligned[i-1] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below 1d HMA + 1w HMA downtrend + volume spike
            elif (close[i] < hma_1d[i] and 
                  hma_1w_aligned[i] < hma_1w_aligned[i-1] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below 1d HMA or volume drop
            if close[i] < hma_1d[i] or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above 1d HMA or volume drop
            if close[i] > hma_1d[i] or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals