#!/usr/bin/env python3
"""
12h_WilliamsAlligator_ElderRay_1wTrend
Hypothesis: Williams Alligator (Jaws/Teeth/Lips) + Elder Ray (Bull/Bear Power) with weekly trend filter
captures trending moves in both bull and bear markets. Long when price > Teeth + Bull Power > 0 + weekly uptrend.
Short when price < Teeth + Bear Power < 0 + weekly downtrend. Uses weekly trend to avoid counter-trend trades.
"""

name = "12h_WilliamsAlligator_ElderRay_1wTrend"
timeframe = "12h"
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

    # Get weekly data for trend filter (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)

    # Williams Alligator: SMAs of median price (HL/2) with different periods
    median_price = (high + low) / 2
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(window=8, min_periods=8).mean().values  # 13-period SMA shifted 8 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(window=5, min_periods=5).mean().values   # 8-period SMA shifted 5 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(window=3, min_periods=3).mean().values   # 5-period SMA shifted 3 bars

    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema21_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above Teeth + Bull Power positive + weekly uptrend
            if close[i] > teeth[i] and bull_power[i] > 0 and close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price below Teeth + Bear Power negative + weekly downtrend
            elif close[i] < teeth[i] and bear_power[i] < 0 and close[i] < ema21_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Teeth or weekly trend turns down
            if close[i] < teeth[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Teeth or weekly trend turns up
            if close[i] > teeth[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals