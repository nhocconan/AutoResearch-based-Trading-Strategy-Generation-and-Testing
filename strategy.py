#!/usr/bin/env python3
"""
6h_Donchian_WeeklyPivot_Trend
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation. 
Weekly pivot provides institutional-level support/resistance, while Donchian captures breakouts. 
Volume confirmation reduces false signals. Works in bull/bear markets by aligning with weekly trend.
Target: 15-25 trades/year per symbol.
"""

name = "6h_Donchian_WeeklyPivot_Trend"
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

    # Get weekly data for pivot and trend (call once before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard floor trader method)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    # R2 = P + (H - L), S2 = P - (H - L)
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Weekly trend: close above/below pivot
    weekly_trend = (weekly_close > pivot).astype(int) * 2 - 1  # 1 for up, -1 for down
    weekly_trend_aligned = align_htf_to_ltf(prices, df_weekly, weekly_trend)

    # Donchian channel (20-period) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any values are NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + weekly uptrend + volume
            if (close[i] > highest_20[i] and 
                weekly_trend_aligned[i] > 0 and 
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + weekly downtrend + volume
            elif (close[i] < lowest_20[i] and 
                  weekly_trend_aligned[i] < 0 and 
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low or weekly trend turns down
            if (close[i] < lowest_20[i] or weekly_trend_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high or weekly trend turns up
            if (close[i] > highest_20[i] or weekly_trend_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals