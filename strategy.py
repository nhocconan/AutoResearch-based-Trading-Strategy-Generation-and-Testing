#!/usr/bin/env python3
# 1d_Pivot_Point_Trend_With_Volume_Filter
# Hypothesis: Use daily pivot points (PP, R1, S1) calculated from weekly high/low/close.
# Long when price closes above R1 with volume > 1.5x 20-day average and weekly trend up (price > weekly EMA20).
# Short when price closes below S1 with volume > 1.5x 20-day average and weekly trend down (price < weekly EMA20).
# Exit when price crosses back through pivot point (PP).
# Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull/bear via weekly trend filter.

name = "1d_Pivot_Point_Trend_With_Volume_Filter"
timeframe = "1d"
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

    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    # Using previous week's values to avoid look-ahead
    pp_1w = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3.0
    r1_1w = 2 * pp_1w - np.roll(low_1w, 1)
    s1_1w = 2 * pp_1w - np.roll(high_1w, 1)
    
    # Align weekly pivot points to daily timeframe
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)

    # Weekly trend filter: price > weekly EMA20 for uptrend, < for downtrend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Volume confirmation: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: close > R1 + volume spike + weekly uptrend (price > weekly EMA20)
            if (close[i] > r1_1w_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: close < S1 + volume spike + weekly downtrend (price < weekly EMA20)
            elif (close[i] < s1_1w_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot point (PP)
            if close[i] < pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above pivot point (PP)
            if close[i] > pp_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals