#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
# Hypothesis: Use weekly (1w) trend with 12h timeframe for tighter entries.
# Enter long when price breaks above daily R1 with volume spike and weekly EMA uptrend.
# Enter short when price breaks below daily S1 with volume spike and weekly EMA downtrend.
# Exit when price returns to the previous day's close (C level).
# Weekly trend filter reduces whipsaw in sideways markets and captures strong trends.
# Target: 15-35 trades/year per symbol.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
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

    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for previous day
    P = (high_1d + low_1d + close_1d) / 3.0
    rng = high_1d - low_1d

    S1 = close_1d - (rng * 1.1 / 12)
    R1 = close_1d + (rng * 1.1 / 12)

    # Align pivot levels to 12h timeframe (use previous day's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, R1)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    # Align previous day's close (C level) for exit
    c_prev = np.roll(close_1d, 1)
    c_prev[0] = np.nan
    c_aligned = align_htf_to_ltf(prices, df_1d, c_prev)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(c_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and weekly EMA uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and weekly EMA downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to previous day's close (C level)
            if close[i] <= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to previous day's close (C level)
            if close[i] >= c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals