#!/usr/bin/env python3
"""
6h_1W_Keltner_Reversal_Trend_Filter
Hypothesis: Trade mean-reversion from weekly Keltner Channel extremes in the direction of 1-day trend.
In ranging markets (common in 2025-2026), price reverts from Keltner bands with high probability.
In trending markets, only take trades aligned with 1-day EMA50 to avoid counter-trend whipsaws.
Uses weekly timeframe for structure (stable, low noise) and 6h for entries.
Target: 20-50 trades/year per symbol, focusing on high-probability mean-reversion spots.
"""

name = "6h_1W_Keltner_Reversal_Trend_Filter"
timeframe = "6h"
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

    # Get weekly data for Keltner Channel (structure)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly Keltner Channel (20, 1.5)
    # Typical Price = (H+L+C)/3
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    atr_period = 20
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])  # align length
    atr_1w = pd.Series(tr_1w).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    ema_tp_1w = pd.Series(tp_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper_1w = ema_tp_1w + 1.5 * atr_1w
    keltner_lower_1w = ema_tp_1w - 1.5 * atr_1w

    # Align Keltner levels to 6h
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper_1w)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower_1w)

    # Get daily trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any data is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches/below lower Keltner AND 1d uptrend (price > EMA50)
            if close[i] <= keltner_lower_aligned[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches/above upper Keltner AND 1d downtrend (price < EMA50)
            elif close[i] >= keltner_upper_aligned[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above EMA_TP (mean reversion complete) or trend fails
            if close[i] >= ema_tp_1w[-1] if i >= len(ema_tp_1w) else ema_tp_1w[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below EMA_TP or trend fails
            if close[i] <= ema_tp_1w[-1] if i >= len(ema_tp_1w) else ema_tp_1w[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals