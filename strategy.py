#!/usr/bin/env python3
# 6h_Pivot_Reversal_12hTrend_Filter
# Hypothesis: Use daily pivot points for mean-reversion entries when price deviates significantly from the pivot,
# filtered by 12h trend direction to align with higher timeframe momentum. Works in both bull and bear markets
# by fading extremes in ranging markets and avoiding counter-trend trades during strong trends.

name = "6h_Pivot_Reversal_12hTrend_Filter"
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

    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # Support 1 = (2 * Pivot) - High
    # Resistance 1 = (2 * Pivot) - Low
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    pivot = (phigh + plow + pclose) / 3.0
    r1 = (2 * pivot) - phigh
    s1 = (2 * pivot) - plow
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)

    # Deviation from pivot as percentage of daily range
    daily_range = phigh - plow
    # Avoid division by zero
    daily_range_safe = np.where(daily_range == 0, 1e-10, daily_range)
    deviation_pct = np.abs(close - pivot_aligned) / daily_range_safe
    
    # Entry when price deviates > 1.5x daily range from pivot (extreme deviation)
    # This captures mean reversion from overextended levels
    extreme_deviation = deviation_pct > 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(extreme_deviation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price significantly below pivot (oversold) and uptrend on 12h
            if close[i] < pivot_aligned[i] and extreme_deviation[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price significantly above pivot (overbought) and downtrend on 12h
            elif close[i] > pivot_aligned[i] and extreme_deviation[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot or trend turns down
            if close[i] >= pivot_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot or trend turns up
            if close[i] <= pivot_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals