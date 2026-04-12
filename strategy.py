#!/usr/bin/env python3
"""
12h_1w_Supertrend_TrendFilter_v1
Hypothesis: Use weekly Supertrend (ATR=10, multiplier=3) as trend filter for 12H timeframe.
Go long when price is above weekly Supertrend and short when below.
Weekly Supertrend adapts to volatility, reducing whipsaws in ranging markets.
Targets 15-25 trades per year to minimize fee drag. Works in bull (follow trend) and bear (counter-trend bounces at trend extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Supertrend_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate ATR
    tr1 = np.subtract(high, low)
    tr2 = np.subtract(high, np.roll(close, 1))
    tr3 = np.subtract(np.roll(close, 1), low)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = np.zeros(n)
    atr[:period-1] = np.nan
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[period-1] = upper_band[period-1]
    direction[period-1] = 1
    
    for i in range(period, n):
        # Supertrend logic
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly data for Supertrend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly Supertrend
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    supertrend_1w, trend_dir_1w = supertrend(weekly_high, weekly_low, weekly_close, 10, 3)
    
    # Align Supertrend to 12h
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    trend_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_dir_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(supertrend_1w_aligned[i]) or 
            np.isnan(trend_dir_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter from weekly Supertrend
        trend_up = trend_dir_1w_aligned[i] == 1
        
        # Entry logic: follow weekly trend
        long_entry = trend_up and close[i] > supertrend_1w_aligned[i]
        short_entry = not trend_up and close[i] < supertrend_1w_aligned[i]
        
        # Exit logic: reverse trend
        long_exit = not trend_up
        short_exit = trend_up
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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