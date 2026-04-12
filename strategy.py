#!/usr/bin/env python3
"""
6h_1d_1w_Liquidity_Reflex_v1
Hypothesis: Price tends to revert from 1-day liquidity pools (overnight gaps, Asian/European/US session opens) 
identified by 1-day high/low clusters, with 1-week trend filter to avoid counter-trend trades.
Works in bull/bear by fading extreme intraday moves that often reverse regardless of trend.
Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_Liquidity_Reflex_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for liquidity levels (previous day high/low clusters)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high and low
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # 1-week trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    weekly_close = df_1w['close'].values
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1-day liquidity levels and weekly trend to 6h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(weekly_close_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: weekly price above/below EMA20
        weekly_trend_up = weekly_close_aligned[i] > weekly_ema20_aligned[i]
        
        # Fade at 1-day liquidity levels with trend filter
        fade_long = low[i] <= prev_low_aligned[i] and weekly_trend_up
        fade_short = high[i] >= prev_high_aligned[i] and not weekly_trend_up
        
        # Exit at midpoint of previous day's range
        midpoint = (prev_high_aligned[i] + prev_low_aligned[i]) / 2
        
        long_exit = close[i] >= midpoint and position == 1
        short_exit = close[i] <= midpoint and position == -1
        
        # Signal logic
        if fade_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif fade_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals