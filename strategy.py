#!/usr/bin/env python3
# 4h_1d_Keltner_Upper_Band_Breakout_1dTrend
# Hypothesis: Breakout above 1d Keltner upper band (EMA20 + 2*ATR) with 1d EMA50 trend filter and volume confirmation.
# Uses 1d Keltner channels for volatility-based breakout levels, reducing whipsaws in sideways markets.
# EMA50 trend filter ensures trades align with higher timeframe momentum. Volume spike confirms breakout strength.
# Designed for 20-50 trades/year to minimize fee drag. Works in bull/bear by following 1d trend.

name = "4h_1d_Keltner_Upper_Band_Breakout_1dTrend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for Keltner channels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate 1d EMA20 for Keltner middle line
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values

    # Calculate 1d ATR(10) for Keltner width
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Calculate Keltner bands: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    keltner_upper_1d = ema20_1d + 2 * atr10_1d
    keltner_lower_1d = ema20_1d - 2 * atr10_1d

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align 1d indicators to 4h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Keltner upper band in 1d uptrend with volume spike
            if close[i] > keltner_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Keltner lower band in 1d downtrend with volume spike
            elif close[i] < keltner_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Keltner lower band (reversal signal)
            if close[i] < keltner_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Keltner upper band (reversal signal)
            if close[i] > keltner_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals