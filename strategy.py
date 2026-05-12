#!/usr/bin/env python3
"""
6h_WeeklyPivot_RangeBound_MeanReversion
Hypothesis: Weekly pivot levels act as strong support/resistance in range-bound markets. 
Price tends to revert to the weekly pivot (mean) after touching weekly S1/S2 or R1/R2 levels.
Uses 6h timeframe with 1-week pivot calculation to capture mean reversion in both bull and bear regimes.
Designed for 10-30 trades/year to minimize fee drag while working in ranging markets.
"""

name = "6h_WeeklyPivot_RangeBound_MeanReversion"
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

    # Get weekly data for pivot calculation (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Use previous week's OHLC for weekly pivot calculation
    prev_week_close = df_1w['close'].shift(1).values
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Calculate weekly pivot and support/resistance levels
    # Standard pivot point calculation
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    
    # Calculate support and resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe with 1-week delay (need previous week's data)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot, additional_delay_bars=1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1, additional_delay_bars=1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2, additional_delay_bars=1)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2, additional_delay_bars=1)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        if np.isnan(pivot_val) or np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(r2_val) or np.isnan(s2_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches or goes below S2, expecting reversion to pivot
            if low[i] <= s2_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches or goes above R2, expecting reversion to pivot
            elif high[i] >= r2_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches or crosses pivot (mean reversion target)
            if close[i] >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches or crosses pivot (mean reversion target)
            if close[i] <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals