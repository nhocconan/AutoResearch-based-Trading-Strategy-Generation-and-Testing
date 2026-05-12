#!/usr/bin/env python3
# 4h_SuperTrend_Volume_Pullback
# Hypothesis: Use SuperTrend (ATR=10, mult=3) for trend direction on 4h, enter on pullbacks to the SuperTrend line during low volatility periods, confirmed by volume spikes (>1.5x 20-period average). Works in bull markets via trend continuation and in bear markets via mean-reversion within the trend. Targets ~25-40 trades/year to minimize fee drag.

name = "4h_SuperTrend_Volume_Pullback"
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

    # Calculate ATR(10) for SuperTrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values

    # SuperTrend calculation
    upper = (high + low) / 2 + 3 * atr
    lower = (high + low) / 2 - 3 * atr

    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend

    supertrend[0] = upper[0]
    direction[0] = 1

    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
            direction[i] = -1

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):
        # Skip if any required value is NaN
        if (np.isnan(supertrend[i]) or np.isnan(direction[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: uptrend (direction=1) + pullback to SuperTrend + volume spike
            if (direction[i] == 1 and 
                close[i] <= supertrend[i] * 1.005 and  # within 0.5% above SuperTrend
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: downtrend (direction=-1) + pullback to SuperTrend + volume spike
            elif (direction[i] == -1 and 
                  close[i] >= supertrend[i] * 0.995 and  # within 0.5% below SuperTrend
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: trend reversal (direction=-1)
            if direction[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: trend reversal (direction=1)
            if direction[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals