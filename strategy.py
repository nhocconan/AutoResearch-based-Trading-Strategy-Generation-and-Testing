#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume2
Hypothesis: On 12h timeframe, price breaking above/below Camarilla R1/S1 levels from previous day indicates momentum continuation. Use daily EMA34 as trend filter (price > EMA34 for long, < EMA34 for short) and daily volume spike (>2x 20-day average) for confirmation. Reduced frequency by raising volume threshold and adding minimum holding period of 3 bars. Targets 20-40 trades/year to avoid fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume2"
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

    # Get daily data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate daily EMA34 for trend
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Calculate Camarilla pivot levels for previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rang = high_1d - low_1d
    r1 = close_1d + rang * 1.1 / 12
    s1 = close_1d - rang * 1.1 / 12

    # Volume confirmation: 2x 20-day average (raised from 1.5x to reduce frequency)
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period

    for i in range(34, n):
        # Get aligned values for current 12h bar
        ema34_a = align_htf_to_ltf(prices, df_1d, ema34)[i]
        r1_a = align_htf_to_ltf(prices, df_1d, r1)[i]
        s1_a = align_htf_to_ltf(prices, df_1d, s1)[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema34_a) or np.isnan(r1_a) or np.isnan(s1_a) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above EMA34 + volume surge
            if (close[i] > r1_a and 
                close[i] > ema34_a and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # SHORT: Price breaks below S1 + price below EMA34 + volume surge
            elif (close[i] < s1_a and 
                  close[i] < ema34_a and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # EXIT LONG: Price falls below EMA34 or S1, but only after minimum 3 bars
            if bars_since_entry >= 3 and (close[i] < ema34_a or close[i] < s1_a):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            bars_since_entry += 1
            # EXIT SHORT: Price rises above EMA34 or R1, but only after minimum 3 bars
            if bars_since_entry >= 3 and (close[i] > ema34_a or close[i] > r1_a):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25

    return signals