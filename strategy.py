#!/usr/bin/env python3
# 1d_WeeklyPivot_MeanReversion
# Hypothesis: Trade reversals at weekly pivot points (R1/S1) on daily timeframe.
# In ranging markets, price tends to revert from weekly support/resistance levels.
# In trending markets, breakouts of these levels can be filtered by weekly trend.
# Combines mean reversion at pivot points with trend filter to work in both bull and bear markets.
# Target: 15-25 trades/year to minimize fee drag while capturing meaningful moves.

name = "1d_WeeklyPivot_MeanReversion"
timeframe = "1d"
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

    # Get weekly data for pivot points and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    # Calculate weekly pivot points (standard formula)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    r2 = pivot + (high_w - low_w)
    s2 = pivot - (high_w - low_w)

    # Align weekly pivot levels to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)

    # Weekly trend filter: EMA21
    ema_21 = pd.Series(close_w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)

    # Daily volume confirmation: volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_confirmed = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price touches S1 with volume confirmation and above weekly EMA21 (uptrend)
            if low[i] <= s1_aligned[i] and volume_confirmed[i] and close[i] > ema_21_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price touches R1 with volume confirmation and below weekly EMA21 (downtrend)
            elif high[i] >= r1_aligned[i] and volume_confirmed[i] and close[i] < ema_21_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches pivot or shows weakness
            if close[i] >= pivot_aligned[i] or low[i] < s1_aligned[i] * 0.995:  # 0.5% buffer
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches pivot or shows strength
            if close[i] <= pivot_aligned[i] or high[i] > r1_aligned[i] * 1.005:  # 0.5% buffer
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals