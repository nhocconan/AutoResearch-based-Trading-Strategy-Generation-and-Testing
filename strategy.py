#!/usr/bin/env python3
# 4h_40_20_EMA_Crossover_With_Volume_Filter
# Hypothesis: Use fast EMA (40) and slow EMA (20) crossover on 4h timeframe with volume confirmation.
# Long when EMA40 crosses above EMA20 with volume spike. Short when EMA40 crosses below EMA20 with volume spike.
# Exit on opposite crossover. Designed to capture medium-term trends with reduced whipsaw via volume filter.
# Volume filter reduces false signals during low-volume consolidations, improving performance in both bull and bear markets.

name = "4h_40_20_EMA_Crossover_With_Volume_Filter"
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

    # Calculate EMAs
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_40 = close_series.ewm(span=40, adjust=False, min_periods=40).mean().values

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):
        # Skip if data is not ready
        if np.isnan(ema_20[i]) or np.isnan(ema_40[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: EMA40 crosses above EMA20 with volume spike
            if ema_40[i] > ema_20[i] and ema_40[i-1] <= ema_20[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: EMA40 crosses below EMA20 with volume spike
            elif ema_40[i] < ema_20[i] and ema_40[i-1] >= ema_20[i-1] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: EMA40 crosses below EMA20
            if ema_40[i] < ema_20[i] and ema_40[i-1] >= ema_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: EMA40 crosses above EMA20
            if ema_40[i] > ema_20[i] and ema_40[i-1] <= ema_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals