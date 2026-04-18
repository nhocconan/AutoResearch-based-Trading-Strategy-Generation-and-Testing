#!/usr/bin/env python3
"""
1d_1w_HighLowBreakout_WeeklyTrend
Hypothesis: On the daily chart, price breaking above the weekly high or below the weekly low indicates momentum continuation. The weekly trend (price above/below weekly EMA20) filters for direction to avoid counter-trend trades. Works in bull markets (breakouts above weekly high in uptrend) and bear markets (breakdowns below weekly low in downtrend). Uses a fixed position size to limit trade frequency and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend and breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20 = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i < 20:
            ema_20[i] = np.nan
        else:
            ema_20[i] = np.mean(close_1w[i-20+1:i+1])
    
    # Weekly high and low for breakout levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly data to daily timeframe (use previous week's values)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: daily close above weekly high AND price above weekly EMA20 (uptrend)
            if close[i] > weekly_high_aligned[i] and close[i] > ema_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily close below weekly low AND price below weekly EMA20 (downtrend)
            elif close[i] < weekly_low_aligned[i] and close[i] < ema_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly EMA20 (trend change)
            if close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly EMA20 (trend change)
            if close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_HighLowBreakout_WeeklyTrend"
timeframe = "1d"
leverage = 1.0