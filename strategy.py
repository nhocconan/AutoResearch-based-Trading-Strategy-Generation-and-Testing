#!/usr/bin/env python3
# 4h_Chaikin_Oscillator_Breakout_1dTrend_Volume
# Hypothesis: Chaikin Oscillator (MACD of ADL) identifies institutional accumulation/distribution.
# Enter long when Chaikin Oscillator crosses above zero with volume spike and 1d EMA50 uptrend.
# Enter short when Chaikin Oscillator crosses below zero with volume spike and 1d EMA50 downtrend.
# Exit when Chaikin Oscillator crosses back across zero (mean reversion).
# Uses 4h timeframe with 1d trend filter to balance trade frequency and win rate.
# Designed to work in both bull (accumulation) and bear (distribution) regimes.
# Target: 20-40 trades/year per symbol.

name = "4h_Chaikin_Oscillator_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    mfm = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    # Money Flow Volume
    mfv = mfm * volume

    # Accumulation Distribution Line
    adl = np.cumsum(mfv)

    # Chaikin Oscillator = EMA(3, ADL) - EMA(10, ADL)
    adl_series = pd.Series(adl)
    ema3 = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10

    # Zero crossover signals
    chaikin_cross_above = np.zeros(n, dtype=bool)
    chaikin_cross_below = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(chaikin[i]) and not np.isnan(chaikin[i-1]):
            chaikin_cross_above[i] = (chaikin[i-1] <= 0) and (chaikin[i] > 0)
            chaikin_cross_below[i] = (chaikin[i-1] >= 0) and (chaikin[i] < 0)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(chaikin[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Chaikin crosses above zero with volume spike and 1d EMA uptrend
            if chaikin_cross_above[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin crosses below zero with volume spike and 1d EMA downtrend
            elif chaikin_cross_below[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin crosses below zero (distribution)
            if chaikin_cross_below[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin crosses above zero (accumulation)
            if chaikin_cross_above[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals