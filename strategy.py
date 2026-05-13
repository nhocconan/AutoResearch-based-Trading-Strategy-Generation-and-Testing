#!/usr/bin/env python3
# 6h_Weekly_Pivot_Breakout_Daily_Trend_Filter
# Hypothesis: Use weekly pivot points as key support/resistance levels. Enter long when price breaks above weekly R1 with daily uptrend filter, enter short when price breaks below weekly S1 with daily downtrend filter.
# Weekly pivots capture institutional levels from prior week; breakouts indicate momentum shifts. Daily trend filter ensures alignment with higher timeframe momentum.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).
# Low frequency due to weekly pivot calculation and breakout confirmation, targeting 15-35 trades/year.

name = "6h_Weekly_Pivot_Breakout_Daily_Trend_Filter"
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

    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly pivot points: (H+L+C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H-L), S2 = P - (H-L)
    # We'll use R1/S1 for breakouts
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly pivots and daily EMA to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above weekly R1 + daily uptrend
            if close[i] > r1_aligned[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S1 + daily downtrend
            elif close[i] < s1_aligned[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below weekly pivot OR trend reversal
            if close[i] < pivot[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above weekly pivot OR trend reversal
            if close[i] > pivot[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals