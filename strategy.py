#!/usr/bin/env python3
# 6h_WeeklyPivot_BullBearPower_Momentum
# Hypothesis: Combines weekly pivot points (from 1w timeframe) with Elder Ray's Bull/Bear Power 
# and momentum filter to capture directional moves. Uses weekly pivot for structure, 
# Bull/Bear Power from daily data for conviction, and 6h ROC for momentum timing. 
# Works in bull markets by buying above weekly pivot with bullish power, 
# and in bear markets by selling below weekly pivot with bearish power. 
# Designed for low trade frequency (target: 50-150/4 years) to minimize fee drag on 6h.

name = "6h_WeeklyPivot_BullBearPower_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Weekly pivot points: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0

    # Get daily data for Bull/Bear Power (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # EMA13 for Bull/Bear Power calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values

    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = ema13_1d - low_1d

    # Align weekly pivot to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)

    # Align daily Bull/Bear Power to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)

    # Momentum filter: 6-period ROC > 0 for long, < 0 for short
    roc_ema = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean()
    roc = (roc_ema / roc_ema.shift(6) - 1) * 100
    roc_values = roc.values
    roc_values[:6] = np.nan  # First 6 values invalid

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required value is NaN
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(roc_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Above weekly pivot + bullish power + positive momentum
            if close[i] > pivot_aligned[i] and bull_power_aligned[i] > 0 and roc_values[i] > 0:
                signals[i] = 0.25
                position = 1
            # SHORT: Below weekly pivot + bearish power + negative momentum
            elif close[i] < pivot_aligned[i] and bear_power_aligned[i] > 0 and roc_values[i] < 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly pivot OR bearish power dominates
            if close[i] < pivot_aligned[i] or bear_power_aligned[i] > bull_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly pivot OR bullish power dominates
            if close[i] > pivot_aligned[i] or bull_power_aligned[i] > bear_power_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals