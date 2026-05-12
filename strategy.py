#!/usr/bin/env python3
"""
6h_WeeklyPivot_Reversal_DailyTrend
Hypothesis: Mean-reversion at weekly pivot support/resistance (R4/S4) with daily trend filter.
In bull markets, buy at S4; in bear markets, sell at R4. Uses daily trend (EMA200) to filter
counter-trend trades in sideways markets. Weekly pivots act as strong support/resistance
levels, especially after extended moves. Designed for low trade frequency (10-30/year).
"""

name = "6h_WeeklyPivot_Reversal_DailyTrend"
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

    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values

    # Calculate weekly pivot points (standard floor trader pivots)
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    r4 = pivot_1w + range_1w * 1.1  # Weekly R4
    s4 = pivot_1w - range_1w * 1.1  # Weekly S4

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Daily EMA200 for trend filter (avoid counter-trend in strong trends)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Pre-calculate weekly pivot alignments (updated only when weekly candle closes)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema200_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches or goes below weekly S4 AND daily trend is up (or neutral)
            if close[i] <= s4_aligned[i] and close[i] >= ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above weekly R4 AND daily trend is down (or neutral)
            elif close[i] >= r4_aligned[i] and close[i] <= ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly pivot (mean reversion target) or reverses at R4
            if close[i] >= pivot_aligned[i] or close[i] >= r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly pivot (mean reversion target) or reverses at S4
            if close[i] <= pivot_aligned[i] or close[i] <= s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals