#!/usr/bin/env python3
# 4h_Chaikin_Momentum_1dTrend_VolumeSurge
# Hypothesis: Chaikin Oscillator (3,10) crossing zero identifies momentum shifts with less whipsaw.
# Enter long when Chaikin crosses above zero with volume surge and 1d EMA50 uptrend.
# Enter short when Chaikin crosses below zero with volume surge and 1d EMA50 downtrend.
# Exit when Chaikin crosses back through zero.
# Uses 4h timeframe with 1d trend filter to balance trade frequency and win rate.
# Designed to work in both bull (buy in uptrend) and bear (sell in downtrend).
# Target: 25-45 trades/year per symbol.

name = "4h_Chaikin_Momentum_1dTrend_VolumeSurge"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Chaikin Oscillator (3,10) on 4h data
    # Money Flow Multiplier
    mfm = np.zeros(n)
    for i in range(n):
        if high[i] == low[i]:
            mfm[i] = 0.0
        else:
            mfm[i] = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])

    # Money Flow Volume
    mfv = mfm * volume

    # Accumulation/Distribution Line
    adl = np.cumsum(mfv)

    # EMA of ADL (3 and 10)
    ema3 = pd.Series(adl).ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = pd.Series(adl).ewm(span=10, adjust=False, min_periods=10).mean().values

    # Chaikin Oscillator
    chaikin = ema3 - ema10

    # Chaikin zero cross signals
    chaikin_cross_above = np.zeros(n, dtype=bool)
    chaikin_cross_below = np.zeros(n, dtype=bool)
    for i in range(1, n):
        chaikin_cross_above[i] = (chaikin[i-1] <= 0) and (chaikin[i] > 0)
        chaikin_cross_below[i] = (chaikin[i-1] >= 0) and (chaikin[i] < 0)

    # Volume surge: current volume > 2.0 x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_surge = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(chaikin[i]) or np.isnan(volume_surge[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Chaikin crosses above zero with volume surge and 1d EMA uptrend
            if chaikin_cross_above[i] and volume_surge[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin crosses below zero with volume surge and 1d EMA downtrend
            elif chaikin_cross_below[i] and volume_surge[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin crosses below zero (momentum loss)
            if chaikin_cross_below[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin crosses above zero (momentum loss)
            if chaikin_cross_above[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals