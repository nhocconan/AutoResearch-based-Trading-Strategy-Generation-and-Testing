#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyTrend_Breakout
# Hypothesis: Weekly pivot levels (R4/S4) define strong support/resistance. 
# Breakouts beyond these levels with daily trend confirmation and volume capture 
# momentum moves. Works in bull/bear via daily trend filter and avoids whipsaws 
# by requiring breakout of significant weekly levels.

name = "6h_WeeklyPivot_DailyTrend_Breakout"
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

    # Weekly high/low for pivot calculation (using 1w data)
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly pivot points from prior week's OHLC
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Standard pivot point calculation
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r4 = pivot + 3 * (weekly_high - weekly_low)  # R4 = PP + 3*(H-L)
    s4 = pivot - 3 * (weekly_high - weekly_low)  # S4 = PP - 3*(H-L)
    
    # Align weekly pivots to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)

    # Daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above weekly R4 + daily EMA uptrend + volume spike
            if (close[i] > r4_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below weekly S4 + daily EMA downtrend + volume spike
            elif (close[i] < s4_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly pivot or volume drop
            if close[i] < (r4_aligned[i] + s4_aligned[i]) / 2 or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly pivot or volume drop
            if close[i] > (r4_aligned[i] + s4_aligned[i]) / 2 or volume[i] < vol_avg_20[i] * 1.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals