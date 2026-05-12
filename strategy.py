#!/usr/bin/env python3
"""
6h_TurtleSoup_Retest_WeeklyPivot
Hypothesis: Trade mean-reversion retests of weekly pivot points (R1/S1) on 6h timeframe when price shows rejection via close back inside weekly pivot range, with confirmation from weekly trend alignment (price above/below weekly EMA50) and volume drying up on the retest. Works in both bull and bear markets by fading extremes and using weekly structure as support/resistance.
Timeframe: 6h
"""

name = "6h_TurtleSoup_Retest_WeeklyPivot"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points and trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate weekly pivot points (using prior week's OHLC)
    ph_w = df_1w['high'].shift(1).values  # prior week high
    pl_w = df_1w['low'].shift(1).values   # prior week low
    pc_w = df_1w['close'].shift(1).values # prior week close
    pw = (ph_w + pl_w + pc_w) / 3.0       # weekly pivot
    r1_w = 2 * pw - pl_w                  # weekly R1
    s1_w = 2 * pw - ph_w                  # weekly S1
    # Align to 6h: weekly pivot values are constant through the week
    pw_aligned = align_htf_to_ltf(prices, df_1w, pw)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)

    # Get weekly data for EMA50 trend filter ONCE before loop
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume drying: current < 0.5x average of last 4 bars (1 day on 6h)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_dry = volume < (0.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):  # Start after EMA50 warmup
        if (np.isnan(pw_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_dry[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: retest of S1 with close back above S1 + price > weekly EMA50 + volume dry
            if (close[i] > s1_w_aligned[i] and low[i] <= s1_w_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and volume_dry[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: retest of R1 with close back below R1 + price < weekly EMA50 + volume dry
            elif (close[i] < r1_w_aligned[i] and high[i] >= r1_w_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and volume_dry[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close < weekly pivot P
            if close[i] < pw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close > weekly pivot P
            if close[i] > pw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals