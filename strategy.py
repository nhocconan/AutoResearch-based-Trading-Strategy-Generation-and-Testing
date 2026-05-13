#!/usr/bin/env python3
# 4h_Elder_Ray_Power_Bull_Bear_Momentum
# Hypothesis: Elder Ray Index measures bull/bear power via EMA and price extremes.
# Long when Bull Power > 0 and Bear Power < 0 with volume confirmation; short when Bear Power > 0 and Bull Power < 0.
# Uses 1d EMA13 as trend filter to avoid counter-trend trades. Designed for low trade frequency (<30/year) to minimize
# fee drag and improve generalization in both bull and bear markets via momentum exhaustion signals.

name = "4h_Elder_Ray_Power_Bull_Bear_Momentum"
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

    # EMA13 for Elder Ray calculation
    def ema(arr, span):
        return pd.Series(arr).ewm(span=span, adjust=False, min_periods=span).mean().values

    ema13 = ema(close, 13)

    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13

    # Get 1d data for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (bulls in control) AND Bear Power < 0 (bears weak) AND price > EMA13 (uptrend) + volume
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema13_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (bears in control) AND Bull Power < 0 (bulls weak) AND price < EMA13 (downtrend) + volume
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  close[i] < ema13_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power turns positive (bears gaining) OR price breaks below EMA13
            if (bear_power[i] >= 0 or close[i] < ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power turns positive (bulls gaining) OR price breaks above EMA13
            if (bull_power[i] >= 0 or close[i] > ema13_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals