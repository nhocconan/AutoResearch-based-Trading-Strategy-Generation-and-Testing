#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1wTrend
# Hypothesis: Use Elder Ray Bull/Bear Power (13-period EMA) on 6h with 1-week trend filter.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Enter long when Bull Power > 0 and Bear Power < 0 (bullish divergence) with 1w EMA uptrend.
# Enter short when Bear Power > 0 and Bull Power < 0 (bearish divergence) with 1w EMA downtrend.
# Exit when Bull Power and Bear Power converge (both near zero) or trend fails.
# Designed to work in both bull (buy bullish divergences) and bear (sell bearish divergences).
# Target: 15-30 trades/year per symbol.

name = "6h_ElderRay_BullBearPower_1wTrend"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 13-period EMA for Elder Ray (6h timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low

    # 1-week EMA21 for trend filter
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Skip if 1w trend data not ready
        if np.isnan(ema21_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (bullish) AND Bear Power < 0 (less bearish) with 1w uptrend
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (bearish) AND Bull Power < 0 (less bullish) with 1w downtrend
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema21_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR trend fails
            if bull_power[i] <= 0 or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns negative OR trend fails
            if bear_power[i] <= 0 or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals