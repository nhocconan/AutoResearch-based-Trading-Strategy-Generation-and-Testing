#!/usr/bin/env python3
# 6h_ElderRay_EMA13_EMA34_Rotation
# Hypothesis: Use Elder Ray (Bull/Bear Power) with EMA13/EMA34 crossover on 6h to detect trend rotation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Enter long when Bull Power > 0 and EMA13 crosses above EMA34 (bullish rotation).
# Enter short when Bear Power > 0 and EMA13 crosses below EMA34 (bearish rotation).
# Exit when EMA13 crosses back in opposite direction.
# Uses EMA13/EMA34 to avoid whipsaw, Elder Ray to filter false crossovers.
# Designed for 6h timeframe with low trade frequency (15-25/year) to minimize fee drag.
# Works in bull/bear via dynamic trend following with momentum confirmation.

name = "6h_ElderRay_EMA13_EMA34_Rotation"
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

    # Calculate EMA13 and EMA34 on 6h
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required value is NaN
        if (np.isnan(ema13[i]) or np.isnan(ema34[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 and EMA13 crosses above EMA34
            if (bull_power[i] > 0 and 
                ema13[i] > ema34[i] and 
                ema13[i-1] <= ema34[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 and EMA13 crosses below EMA34
            elif (bear_power[i] > 0 and 
                  ema13[i] < ema34[i] and 
                  ema13[i-1] >= ema34[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: EMA13 crosses below EMA34
            if ema13[i] < ema34[i] and ema13[i-1] >= ema34[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: EMA13 crosses above EMA34
            if ema13[i] > ema34[i] and ema13[i-1] <= ema34[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals