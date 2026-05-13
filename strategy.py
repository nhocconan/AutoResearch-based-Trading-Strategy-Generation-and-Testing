#!/usr/bin/env python3
# 6h_AngleBasedTrend_1dVolatilityFilter
# Hypothesis: Use the angle (rate of change) of a 6h EMA20 to determine trend strength and direction,
# combined with 1d volatility regime (ATR ratio) to filter entries. Enter long when EMA20 angle is
# positive and steep (>0.05%) and volatility is contracting (ATR7/ATR30 < 0.8), short when angle is
# negative and steep (<-0.05%) with contracting volatility. Exit when angle flattens or volatility expands.
# Designed to capture strong trending moves while avoiding choppy markets, targeting 15-30 trades/year.

name = "6h_AngleBasedTrend_1dVolatilityFilter"
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

    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR(30) and ATR(7) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    atr30_1d = pd.Series(tr1).rolling(window=30, min_periods=30).mean().values
    atr7_1d = pd.Series(tr1).rolling(window=7, min_periods=7).mean().values
    atr_ratio_1d = atr7_1d / atr30_1d  # <1 = contracting volatility
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)

    # 6h EMA20 for trend angle calculation
    ema20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # Calculate angle as % change over 3 periods: (EMA[t] - EMA[t-3]) / EMA[t-3] * 100
    ema20_delayed = np.roll(ema20_6h, 3)
    ema20_delayed[:3] = np.nan
    angle_ema20 = (ema20_6h - ema20_delayed) / ema20_delayed * 100  # in percent

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(angle_ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Steep positive angle + contracting volatility
            if angle_ema20[i] > 0.05 and atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.25
                position = 1
            # SHORT: Steep negative angle + contracting volatility
            elif angle_ema20[i] < -0.05 and atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Angle flattens or volatility expands
            if angle_ema20[i] < 0.0 or atr_ratio_1d_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Angle flattens or volatility expands
            if angle_ema20[i] > 0.0 or atr_ratio_1d_aligned[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals