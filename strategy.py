#!/usr/bin/env python3
# 6h_Chaikin_Oscillator_Momentum_With_Trend_Filter
# Hypothesis: Chaikin Oscillator (3,10) crossing zero indicates short-term momentum shifts in accumulation/distribution.
# Confirmed by daily trend (EMA50) to avoid counter-trend trades. Volume-weighted ensures institutional participation.
# Designed for 6h timeframe with low trade frequency (<30/year) to minimize fee drag.
# Works in both bull and bear markets by following the higher timeframe trend.

name = "6h_Chaikin_Oscillator_Momentum_With_Trend_Filter"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    # Avoid division by zero
    hl_range = high - low
    mf_multiplier = np.where(hl_range != 0, ((close - low) - (high - close)) / hl_range, 0.0)
    money_flow_volume = mf_multiplier * volume

    # Chaikin Oscillator = EMA(3) of MFV - EMA(10) of MFV
    mfv_series = pd.Series(money_flow_volume)
    ema3 = mfv_series.ewm(span=3, adjust=False, min_periods=3).mean()
    ema10 = mfv_series.ewm(span=10, adjust=False, min_periods=10).mean()
    chaikin_osc = (ema3 - ema10).values

    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(10, n):  # Start after EMA10 needs 10 bars
        # Skip if any required data is NaN
        if (np.isnan(chaikin_osc[i]) or np.isnan(chaikin_osc[i-1]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Chaikin Oscillator crosses above zero with uptrend
            if (chaikin_osc[i-1] <= 0 and chaikin_osc[i] > 0 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Chaikin Oscillator crosses below zero with downtrend
            elif (chaikin_osc[i-1] >= 0 and chaikin_osc[i] < 0 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Chaikin Oscillator crosses below zero
            if chaikin_osc[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Chaikin Oscillator crosses above zero
            if chaikin_osc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals