#!/usr/bin/env python3
# 4h_WilliamsVixFix_TrendFollow
# Hypothesis: Williams Vix Fix volatility spike with 1d EMA trend filter and volume confirmation.
# The Vix Fix identifies panic selling/buying climaxes; combined with trend filter avoids counter-trend trades.
# Volume spike confirms institutional participation. Designed for 75-200 total trades over 4 years (19-50/year).
# Works in bull/bear by following 1d trend direction. Uses Williams Vix Fix formula: (Highest Close - Low) / (Highest Close) * 100.

name = "4h_WilliamsVixFix_TrendFollow"
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

    # Get 4h data for price action and Vix Fix calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 22:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)

    # Calculate Williams Vix Fix: (Highest Close in period - Low) / (Highest Close) * 100
    # Using 22-period lookback to match common Vix Fix settings
    highest_close = pd.Series(close_4h).rolling(window=22, min_periods=22).max().values
    vixfix = (highest_close - low_4h) / highest_close * 100
    # Vix Fix values typically range 0-100, higher = more fear/volatility

    # Calculate EMA of Vix Fix for signal smoothing
    vixfix_ema = pd.Series(vixfix).ewm(span=9, adjust=False, min_periods=9).mean().values

    # Align Vix Fix EMA to 4h timeframe
    vixfix_ema_aligned = align_htf_to_ltf(prices, df_4h, vixfix_ema)

    # Calculate 4h EMA20 for trend confirmation (additional filter)
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)

    # Calculate 4h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.8  # Require 1.8x average volume for significance

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(22, n):
        # Skip if any required data is NaN
        if (np.isnan(vixfix_ema_aligned[i]) or np.isnan(ema20_1d_aligned[i]) or
            np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Vix Fix spike (fear) in 1d uptrend with volume spike and price above EMA20
            if (vixfix_ema_aligned[i] > 25 and  # Significant fear spike
                close[i] > ema20_1d_aligned[i] and  # 1d uptrend
                close[i] > ema20_4h_aligned[i] and  # 4h price above EMA20
                volume[i] > volume_sma20[i]):       # Volume confirmation
                signals[i] = 0.25
                position = 1
            # SHORT: Vix Fix spike (fear) in 1d downtrend with volume spike and price below EMA20
            elif (vixfix_ema_aligned[i] > 25 and   # Significant fear spike
                  close[i] < ema20_1d_aligned[i] and  # 1d downtrend
                  close[i] < ema20_4h_aligned[i] and  # 4h price below EMA20
                  volume[i] > volume_sma20[i]):       # Volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Vix Fix normalizes OR price closes below 4h EMA20 (trend change)
            if (vixfix_ema_aligned[i] < 15 or  # Fear subsided
                close[i] < ema20_4h_aligned[i]):  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Vix Fix normalizes OR price closes above 4h EMA20 (trend change)
            if (vixfix_ema_aligned[i] < 15 or  # Fear subsided
                close[i] > ema20_4h_aligned[i]):  # Trend broken
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals