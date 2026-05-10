#!/usr/bin/env python3
# 6h_WeeklyPivot_Reversal_1dTrend_Filter
# Hypothesis: Uses weekly pivot levels (S1/S2/R1/R2) from previous week with 1d trend filter.
# In ranging markets (common in 2025+), price tends to revert from weekly S1/S2 and R1/R2.
# In trending markets, we filter by 1d EMA to only take trades in trend direction.
# This combines mean reversion at key weekly levels with trend filtering to work in both bull/bear.
# Position size 0.25 to manage drawdown. Target: 20-40 trades/year.

name = "6h_WeeklyPivot_Reversal_1dTrend_Filter"
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
    
    # Get weekly data for pivot levels (using previous week's data)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week's OHLC
    prev_week_high = df_w['high'].shift(1).values
    prev_week_low = df_w['low'].shift(1).values
    prev_week_close = df_w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    
    # Calculate 1d EMA for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # Warmup for 1d EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price touches or goes below S1/S2 with 1d uptrend (mean reversion in uptrend)
            # or price touches or goes below S2 with any trend (stronger signal)
            if (close[i] <= s1_aligned[i] and uptrend) or close[i] <= s2_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches or goes above R1/R2 with 1d downtrend (mean reversion in downtrend)
            # or price touches or goes above R2 with any trend (stronger signal)
            elif (close[i] >= r1_aligned[i] and downtrend) or close[i] >= r2_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot or R1, or trend turns down
            if close[i] >= pivot[i] or close[i] >= r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot or S1, or trend turns up
            if close[i] <= pivot[i] or close[i] <= s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals