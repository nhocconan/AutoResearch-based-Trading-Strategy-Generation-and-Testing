#!/usr/bin/env python3
"""
6h_ElderRay_Power_1wTrend
Hypothesis: Elder Ray (Bull/Bear Power) with 1-week trend filter on 6h timeframe.
Bull Power = High - EMA13, Bear Power = EMA13 - Low. 
Long when Bull Power > 0 and 1w uptrend; Short when Bear Power > 0 and 1w downtrend.
Uses 13-period EMA for responsiveness. Weekly trend avoids whipsaws in sideways markets.
Designed for low trade frequency (~20-40/year) to minimize fee impact. Works in bull/bear via trend filter.
"""

name = "6h_ElderRay_Power_1wTrend"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter (slow enough to avoid whipsaws)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Get aligned weekly trend (only uses completed weekly bars)
        ema34_aligned = ema34_1w_aligned[i]

        # Skip if any required data is NaN
        if np.isnan(ema34_aligned) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power positive (strong buying pressure) + weekly uptrend
            if bull_power[i] > 0 and close[i] > ema34_aligned:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power positive (strong selling pressure) + weekly downtrend
            elif bear_power[i] > 0 and close[i] < ema34_aligned:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power becomes positive (selling pressure takes over) OR weekly trend turns down
            if bear_power[i] > 0 or close[i] < ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power becomes positive (buying pressure takes over) OR weekly trend turns up
            if bull_power[i] > 0 or close[i] > ema34_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals