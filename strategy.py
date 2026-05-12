#!/usr/bin/env python3
# 6h_Retracement_to_Pivot_with_OrderFlow
# Hypothesis: On 6h timeframe, price often retraces to key pivot levels (PP, S1, R1) before continuing the trend.
# We use daily pivot points calculated from prior day's OHLC. Enter long when price retraces to S1 or PP in an uptrend (price > 200 EMA),
# and short when price retraces to R1 or PP in a downtrend (price < 200 EMA). Volume confirmation filters out false retraces.
# Exit when price reaches the opposite pivot level or shows exhaustion (volume dry-up). Designed for low frequency (15-35 trades/year).
# Works in bull markets by buying dips in uptrends, and in bear markets by selling rallies in downtrends.

name = "6h_Retracement_to_Pivot_with_OrderFlow"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get daily data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points (using prior day's OHLC to avoid look-ahead)
    # Pivot Point (PP) = (High + Low + Close) / 3
    # Resistance 1 (R1) = (2 * PP) - Low
    # Support 1 (S1) = (2 * PP) - High
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = (2 * pp_1d) - low_1d
    s1_1d = (2 * pp_1d) - high_1d

    # Align pivot levels to 6h timeframe (use prior day's levels for current day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # Get 200 EMA for trend filter (daily)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Volume filter: 1.5x 20-period SMA on 6h
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema200_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price retraces to S1 or PP in uptrend with volume
            retrace_to_support = (abs(close[i] - s1_aligned[i]) < 0.001 * close[i]) or (abs(close[i] - pp_aligned[i]) < 0.001 * close[i])
            if retrace_to_support and close[i] > ema200_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price retraces to R1 or PP in downtrend with volume
            elif (abs(close[i] - r1_aligned[i]) < 0.001 * close[i]) or (abs(close[i] - pp_aligned[i]) < 0.001 * close[i]):
                if close[i] < ema200_aligned[i] and volume[i] > volume_sma20[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches R1 or shows weakness (low volume near PP)
            if (abs(close[i] - r1_aligned[i]) < 0.001 * close[i]) or \
               (abs(close[i] - pp_aligned[i]) < 0.001 * close[i] and volume[i] < volume_sma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches S1 or shows weakness (low volume near PP)
            if (abs(close[i] - s1_aligned[i]) < 0.001 * close[i]) or \
               (abs(close[i] - pp_aligned[i]) < 0.001 * close[i] and volume[i] < volume_sma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals