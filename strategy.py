#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_Trend_Filter
Hypothesis: Trade breakouts of weekly R1/S1 pivot levels on 6h with 1d trend filter to avoid counter-trend trades.
Long when price breaks above weekly R1 with 1d uptrend; short when breaks below weekly S1 with 1d downtrend.
Uses weekly pivots (calculated from prior week's OHLC) for structural levels and 1d EMA50 for trend filter.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: 1d trend filter avoids counter-trend trades, weekly pivots provide significant support/resistance.
"""

name = "6h_WeeklyPivot_R1_S1_Breakout_Trend_Filter"
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
    
    # Get weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivots from prior week's OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # R1 = (2 * PP) - L
    r1 = (2 * pp) - weekly_low
    # S1 = (2 * PP) - H
    s1 = (2 * pp) - weekly_high
    
    # Align weekly pivots to 6h timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with 1d uptrend (price > EMA50)
            if close[i] > r1_aligned[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with 1d downtrend (price < EMA50)
            elif close[i] < s1_aligned[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly S1 OR 1d trend turns down
            if close[i] < s1_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly R1 OR 1d trend turns up
            if close[i] > r1_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals