#!/usr/bin/env python3
# 12h_WeeklyPivot_R1S1_Breakout_WeekTrend_Volume
# Hypothesis: Use weekly Camarilla pivot levels (R1/S1) as support/resistance on 12h timeframe.
# Enter long when price breaks above weekly R1 with volume spike and weekly close above weekly open (bullish week).
# Enter short when price breaks below weekly S1 with volume spike and weekly close below weekly open (bearish week).
# Exit when price returns to the weekly close (C level).
# Designed to work in bull markets (buy breakouts in bullish weeks) and bear markets (sell breakdowns in bearish weeks).
# Target: 15-30 trades/year per symbol.

name = "12h_WeeklyPivot_R1S1_Breakout_WeekTrend_Volume"
timeframe = "12h"
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
    volume = prices['volume'].values

    # Get weekly data for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values

    # Calculate weekly Camarilla pivot levels for previous week
    # P = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # R1 = C + (Range * 1.1 / 12)
    P = (high_1w + low_1w + close_1w) / 3.0
    rng = high_1w - low_1w

    S1 = close_1w - (rng * 1.1 / 12)
    R1 = close_1w + (rng * 1.1 / 12)

    # Align pivot levels to 12h timeframe (use previous week's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, R1)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Weekly trend: bullish if close > open, bearish if close < open
    weekly_bullish = close_1w > open_1w
    weekly_bearish = close_1w < open_1w
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish)

    # Align previous week's close (C level) for exit
    c_prev = np.roll(close_1w, 1)
    c_prev[0] = np.nan
    c_aligned = align_htf_to_ltf(prices, df_1w, c_prev)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(weekly_bullish_aligned[i]) or
            np.isnan(weekly_bearish_aligned[i]) or np.isnan(c_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above weekly R1 with volume spike and bullish week
            if close[i] > r1_aligned[i] and volume_spike[i] and weekly_bullish_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below weekly S1 with volume spike and bearish week
            elif close[i] < s1_aligned[i] and volume_spike[i] and weekly_bearish_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to previous week's close (C level)
            if close[i] <= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to previous week's close (C level)
            if close[i] >= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals