#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Filter
# Hypothesis: Elder Ray Index (Bull Power/Bear Power) with 1-day trend filter.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Trades in direction of 1d EMA50 trend.
# Uses 13-period EMA for Elder Ray calculation. Low trade frequency expected due to trend filter requirement.
# Works in bull markets (long when Bull Power > 0 and uptrend) and bear markets (short when Bear Power < 0 and downtrend).

name = "6h_ElderRay_BullBearPower_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate EMA13 for Elder Ray (using daily close)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Calculate Bull Power and Bear Power
    bull_power = high_1d - ema13_1d  # High - EMA13
    bear_power = ema13_1d - low_1d   # EMA13 - Low

    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Align all 1d indicators to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (bullish momentum) + price above EMA50 (uptrend) + volume confirmation
            if (bull_power_aligned[i] > 0 and
                close[i] > ema50_aligned[i] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bearish momentum) + price below EMA50 (downtrend) + volume confirmation
            elif (bear_power_aligned[i] < 0 and
                  close[i] < ema50_aligned[i] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power >= 0 (loss of bullish momentum) OR price below EMA50 (trend change)
            if bear_power_aligned[i] >= 0 or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power <= 0 (loss of bearish momentum) OR price above EMA50 (trend change)
            if bull_power_aligned[i] <= 0 or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals