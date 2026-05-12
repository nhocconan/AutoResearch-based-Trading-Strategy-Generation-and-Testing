#!/usr/bin/env python3
"""
6h_ElderRay_Power_1wTrend
Hypothesis: Use weekly Elder Ray power (bull/bear) for trend direction and 6-hour
price crossing EMA13 for entry. Exit when power reverses. Weekly trend filter
avoids counter-trend trades, improving performance in both bull and bear markets.
Target: ~20-40 trades/year per symbol.
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
    volume = prices['volume'].values

    # Get weekly data for Elder Ray power
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Calculate EMA13 for weekly trend and power
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = weekly high - EMA13
    bull_power = high_1w - ema13_1w
    # Bear Power = weekly low - EMA13
    bear_power = low_1w - ema13_1w

    # Align weekly indicators to 6h timeframe
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)

    # 6h EMA13 for entry signal
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        # Get aligned values for current 6h bar (only uses completed weekly bars)
        ema13_w = ema13_1w_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema13_h = ema13_6h[i]

        # Skip if any required data is NaN
        if (np.isnan(ema13_w) or np.isnan(bull) or np.isnan(bear) or np.isnan(ema13_h)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Weekly bull power positive AND price crosses above 6h EMA13
            if bull > 0 and close[i] > ema13_h:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly bear power negative AND price crosses below 6h EMA13
            elif bear < 0 and close[i] < ema13_h:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly bull power turns negative
            if bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly bear power turns positive
            if bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals