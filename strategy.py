#!/usr/bin/env python3
# 12h_1W_Camarilla_Pivot_Volume
# Hypothesis: Long when price breaks above weekly Camarilla R4 with volume spike and 1d EMA uptrend; short when breaks below S4 with volume spike and 1d EMA downtrend.
# Exit when price reverts to weekly pivot point (PP). Uses weekly structure to avoid overtrading and capture strong breakouts in trending markets.
# Target: 15-25 trades/year on 12h to minimize fee drift while capturing strong moves in both bull and bear markets.

name = "12h_1W_Camarilla_Pivot_Volume"
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

    # Get weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly Camarilla levels (based on prior week)
    # Pivot Point (PP) = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1 / 2
    # S4 = PP - (H - L) * 1.1 / 2
    pp = (high_1w + low_1w + close_1w) / 3.0
    r4 = pp + (high_1w - low_1w) * 1.1 / 2.0
    s4 = pp - (high_1w - low_1w) * 1.1 / 2.0

    # Shift to use prior week's levels (no look-ahead)
    pp = np.roll(pp, 1)
    r4 = np.roll(r4, 1)
    s4 = np.roll(s4, 1)
    pp[0] = np.nan
    r4[0] = np.nan
    s4[0] = np.nan

    # Align weekly levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)

    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2.0x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R4 + price > 1d EMA34 + volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S4 + price < 1d EMA34 + volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_30[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below PP (mean reversion to pivot)
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above PP
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals