#!/usr/bin/env python3
# 6h_1D_ElderRay_BullPower_With_1D_Trend
# Hypothesis: Elder Ray Bull Power (close - EMA13) on 1d confirms institutional buying pressure.
# Enter long when Bull Power turns positive with price above EMA20 on 6h; exit when Bull Power turns negative.
# Short when Bear Power (EMA13 - close) turns positive with price below EMA20 on 6h; exit when Bear Power turns negative.
# Uses 1d for Elder Ray (institutional force) and 6s for entry/exit timing.
# Works in bull markets (sustained Bull Power > 0) and bear markets (sustained Bear Power > 0) by following institutional momentum.

name = "6h_1D_ElderRay_BullPower_With_1D_Trend"
timeframe = "6h"
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

    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)

    # Calculate EMA13 on 1d close for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power = close - EMA13, Bear Power = EMA13 - close
    bull_power = close_1d - ema13_1d
    bear_power = ema13_1d - close_1d

    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)

    # Calculate EMA20 on 6h close for trend filter
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema20_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power turns positive AND price above EMA20
            if bull_power_aligned[i] > 0 and bull_power_aligned[i-1] <= 0 and close[i] > ema20_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power turns positive AND price below EMA20
            elif bear_power_aligned[i] > 0 and bear_power_aligned[i-1] <= 0 and close[i] < ema20_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative
            if bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative
            if bear_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals