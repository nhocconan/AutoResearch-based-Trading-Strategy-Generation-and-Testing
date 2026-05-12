#!/usr/bin/env python3
"""
4h_TRIX_Trend_With_Volume_Confirmation
Hypothesis: Use TRIX (triple exponential moving average) as a trend-following indicator on 4h timeframe, entering long when TRIX turns positive and short when negative, confirmed by volume spikes. Works in both bull and bear markets by capturing momentum shifts. TRIX filters out noise and volatility, while volume confirmation ensures institutional participation. Exits when TRIX crosses zero or momentum weakens.
Timeframe: 4h
"""

name = "4h_TRIX_Trend_With_Volume_Confirmation"
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

    # TRIX: triple EMA of price, then rate of change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (pd.Series(ema3).pct_change(1))
    trix_values = trix.values

    # Volume spike: current > 1.8x average of last 6 bars (1 day on 4h)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(40, n):  # Start after TRIX warmup
        if np.isnan(trix[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: TRIX crosses above zero + volume spike
            if trix[i] > 0 and trix[i-1] <= 0 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + volume spike
            elif trix[i] < 0 and trix[i-1] >= 0 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals