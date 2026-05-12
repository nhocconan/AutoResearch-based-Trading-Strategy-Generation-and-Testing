#!/usr/bin/env python3
# 4h_Donchian_Breakout_VolumeTrend_Filter_v1
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x SMA20) and trend filter (price > EMA50).
# Works in bull markets via breakout momentum and in bear via short breakdowns.
# Volume filter reduces false breakouts; EMA50 ensures trend alignment.
# Target: 20-50 trades/year to avoid fee drag.

name = "4h_Donchian_Breakout_VolumeTrend_Filter_v1"
timeframe = "4h"
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

    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: 1.5x SMA20
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema50[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above Donchian high + volume spike + uptrend
            if (close[i] > high_roll[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below Donchian low + volume spike + downtrend
            elif (close[i] < low_roll[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian low
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals