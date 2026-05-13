#!/usr/bin/env python3
# 12h_WeeklyPivot_RangeBreakout_1dTrend
# Hypothesis: Use weekly pivot points (PP, S1, S2, R1, R2) for mean reversion in range-bound markets.
# In low volatility (choppy) regimes, price tends to revert from S1/R1 toward the pivot.
# Add 1d EMA trend filter to avoid trading against the dominant trend.
# Weekly pivots are calculated from prior week's OHLC and remain static through the week.
# This strategy targets 15-25 trades per year on 12h timeframe, avoiding excessive churn.

name = "12h_WeeklyPivot_RangeBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    # Calculate weekly pivot points: PP = (H+L+C)/3, S1 = 2*PP - H, R1 = 2*PP - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    pp = (high_1w + low_1w + close_1w) / 3.0
    s1 = 2 * pp - high_1w
    r1 = 2 * pp - low_1w

    # Align weekly pivots to 12h timeframe (no look-ahead, uses prior week's values)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)

    # Choppy market filter: Choppiness Index (CI) > 61.8 = range-bound
    # CI = 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Smooth TR with SMA
        atr[period-1] = np.mean(tr[1:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Calculate Chop
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        sum_atr = np.zeros_like(close)
        for i in range(period-1, len(close)):
            sum_atr[i] = np.sum(atr[i-period+1:i+1])
        range_val = max_high - min_low
        chop = np.full_like(close, 50.0, dtype=float)
        mask = range_val > 0
        chop[mask] = 100 * np.log10(sum_atr[mask] / range_val[mask]) / np.log10(period)
        return chop

    chop = choppiness_index(high, low, close, 14)
    chop_filter = chop > 61.8  # Range-bound market

    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(chop_filter[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price near S1 in choppy market and 1d uptrend
            if chop_filter[i] and low[i] <= s1_aligned[i] * 1.001 and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price near R1 in choppy market and 1d downtrend
            elif chop_filter[i] and high[i] >= r1_aligned[i] * 0.999 and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reaches PP or chop ends
            if close[i] >= pp_aligned[i] * 0.999 or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price reaches PP or chop ends
            if close[i] <= pp_aligned[i] * 1.001 or not chop_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals