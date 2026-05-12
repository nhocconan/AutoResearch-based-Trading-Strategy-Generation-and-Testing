#!/usr/bin/env python3
# 4h_KAMA_Adaptive_Trend_Signal
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) with adaptive smoothing captures trend changes effectively in both bull and bear markets.
# KAMA reduces whipsaw by adjusting sensitivity based on market noise (efficiency ratio). Combined with volume confirmation and 1d EMA50 trend filter.
# Designed for low trade frequency by requiring alignment of KAMA direction change, volume spike, and higher timeframe trend.
# Exit on opposite KAMA crossover to avoid overtrading.

name = "4h_KAMA_Adaptive_Trend_Signal"
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

    # Get 4h data for KAMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    close_4h = df_4h['close'].values

    # Calculate Efficiency Ratio (ER) and Smoothing Constants for KAMA
    change = np.abs(np.subtract(close_4h[10:], close_4h[:-10]))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_4h)), axis=0)  # Will be calculated properly below

    # Proper ER calculation: need 10-period volatility
    er = np.full_like(close_4h, np.nan)
    for i in range(10, len(close_4h)):
        price_change = np.abs(close_4h[i] - close_4h[i-10])
        price_volatility = np.sum(np.abs(np.diff(close_4h[i-10:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0

    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    sc[np.isnan(sc)] = 0

    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])

    # Align KAMA to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate volume spike threshold (1.5x 20-period SMA)
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA crosses above price (bullish shift) in uptrend with volume spike
            if (kama_aligned[i] > close[i] and kama_aligned[i-1] <= close[i-1] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA crosses below price (bearish shift) in downtrend with volume spike
            elif (kama_aligned[i] < close[i] and kama_aligned[i-1] >= close[i-1] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA crosses below price (bearish shift)
            if kama_aligned[i] < close[i] and kama_aligned[i-1] >= close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA crosses above price (bullish shift)
            if kama_aligned[i] > close[i] and kama_aligned[i-1] <= close[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals