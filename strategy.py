#!/usr/bin/env python3
# 6H_WeeklyPivot_Direction_1dTrend_Filter
# Hypothesis: Use weekly pivot points (from prior week) as dynamic support/resistance.
# Enter long when price crosses above weekly pivot + price > 1d EMA50 (bullish trend).
# Enter short when price crosses below weekly pivot + price < 1d EMA50 (bearish trend).
# Exit when price crosses back below/above weekly pivot or trend changes.
# Uses weekly trend filter to avoid counter-trend trades, works in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).

name = "6H_WeeklyPivot_Direction_1dTrend_Filter"
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
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for weekly pivot calculation (prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Pivot = (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    
    # Align weekly pivot to 6h timeframe (use prior week's pivot)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 2)  # Warmup for EMA and pivot (need at least 2 weeks)
    
    for i in range(start_idx, n):
        if np.isnan(ema_1d_aligned[i]) or np.isnan(pivot_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Long entry: price crosses above weekly pivot + above 1d EMA50
            if close[i] > pivot_aligned[i] and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below weekly pivot + below 1d EMA50
            elif close[i] < pivot_aligned[i] and price_below_ema:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below weekly pivot OR trend turns bearish
            if close[i] < pivot_aligned[i] or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above weekly pivot OR trend turns bullish
            if close[i] > pivot_aligned[i] or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals