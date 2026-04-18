#!/usr/bin/env python3
"""
6h_WeeklyPivot_Trend_Follow
Hypothesis: Trend-following strategy using weekly pivot point direction on 6h timeframe.
In 6h bars, enter long when price is above weekly pivot and 6h EMA20 rising, short when below pivot and EMA20 falling.
Weekly pivot provides macro trend filter; EMA20 provides entry timing. Works in bull/bear by adapting pivot levels.
Designed for low trade frequency (~15-25/year) with high win rate via confluence of weekly structure and trend.
"""

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
    
    # Weekly pivot point calculation (using weekly OHLC)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot point = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Align weekly pivot to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # EMA20 on 6h for trend timing
    k = 2 / (20 + 1)
    ema20 = np.full(n, np.nan)
    for i in range(20, n):
        if i == 20:
            ema20[i] = np.mean(close[0:21])
        else:
            ema20[i] = close[i] * k + ema20[i-1] * (1 - k)
    
    # EMA20 slope for trend direction
    ema20_slope = np.full(n, np.nan)
    for i in range(21, n):
        if not np.isnan(ema20[i]) and not np.isnan(ema20[i-1]):
            ema20_slope[i] = ema20[i] - ema20[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for EMA20
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(ema20[i]) or np.isnan(ema20_slope[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly pivot AND EMA20 rising
            if close[i] > pivot_aligned[i] and ema20_slope[i] > 0:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly pivot AND EMA20 falling
            elif close[i] < pivot_aligned[i] and ema20_slope[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below weekly pivot OR EMA20 turns down
            if close[i] < pivot_aligned[i] or ema20_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above weekly pivot OR EMA20 turns up
            if close[i] > pivot_aligned[i] or ema20_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Trend_Follow"
timeframe = "6h"
leverage = 1.0