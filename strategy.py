#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Pivot_Breakout_Trend
Hypothesis: Uses weekly and daily pivots for breakout entries on 12h timeframe with trend filter.
Designed for low frequency (12-37 trades/year) to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Pivot_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly trend filter: 20-period EMA
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly pivots for context
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_pivot_high = close_1w + 1.1 * (high_1w - low_1w)  # Weekly R3
    weekly_pivot_low = close_1w - 1.1 * (high_1w - low_1w)   # Weekly S3
    weekly_pivot_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_high)
    weekly_pivot_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_low)
    
    # Daily pivots for entry/exit
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    daily_pivot_high = close_1d + 1.1 * (high_1d - low_1d)  # Daily R3
    daily_pivot_low = close_1d - 1.1 * (high_1d - low_1d)   # Daily S3
    daily_pivot_high_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_high)
    daily_pivot_low_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(weekly_pivot_high_aligned[i]) or 
            np.isnan(weekly_pivot_low_aligned[i]) or np.isnan(daily_pivot_high_aligned[i]) or 
            np.isnan(daily_pivot_low_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price relative to weekly EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions: require both weekly and daily alignment
        # Long: price breaks above BOTH weekly and daily R3 in uptrend
        # Short: price breaks below BOTH weekly and daily S3 in downtrend
        long_breakout = (close[i] > weekly_pivot_high_aligned[i]) and (close[i] > daily_pivot_high_aligned[i])
        short_breakout = (close[i] < weekly_pivot_low_aligned[i]) and (close[i] < daily_pivot_low_aligned[i])
        
        # Entry conditions
        long_entry = long_breakout and uptrend
        short_entry = short_breakout and downtrend
        
        # Exit conditions: return to opposite pivot level or trend reversal
        long_exit = (close[i] < daily_pivot_low_aligned[i]) or (not uptrend)
        short_exit = (close[i] > daily_pivot_high_aligned[i]) or (not downtrend)
        
        # Priority: entry > exit > hold
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
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals