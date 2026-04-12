#!/usr/bin/env python3
"""
1d_1w_Camarilla_Trend_Follower_v1
Hypothesis: Weekly trend (price above/below weekly SMA20) filters daily entries.
Go long when daily price breaks above daily R4 AND weekly trend is up.
Go short when daily price breaks below daily S4 AND weekly trend is down.
Uses weekly trend to avoid counter-trend trades in strong trends, reducing whipsaws.
Targets 20-50 total trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Trend_Follower_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly SMA20 for trend
    weekly_close = df_1w['close'].values
    weekly_sma20 = np.full(len(weekly_close), np.nan)
    for i in range(20, len(weekly_close)):
        weekly_sma20[i] = np.mean(weekly_close[i-20:i])
    
    weekly_trend_up = weekly_sma20 > 0  # Will be filled properly after alignment
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r4 = prev_close + range_ * 1.1 / 2
    camarilla_s4 = prev_close - range_ * 1.1 / 2
    
    # Handle invalid ranges
    valid_range = range_ > 0
    camarilla_r4 = np.where(valid_range, camarilla_r4, np.nan)
    camarilla_s4 = np.where(valid_range, camarilla_s4, np.nan)
    
    # Align to daily timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    weekly_sma20_aligned = align_htf_to_ltf(prices, df_1w, weekly_sma20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(weekly_sma20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: weekly price above/below SMA20
        weekly_close_price = df_1w['close'].values
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close_price)
        weekly_trend = weekly_close_aligned[i] > weekly_sma20_aligned[i]
        
        # Breakout conditions
        long_breakout = high[i] > camarilla_r4_aligned[i] and weekly_trend
        short_breakout = low[i] < camarilla_s4_aligned[i] and not weekly_trend
        
        # Exit conditions: return to Camarilla midpoint
        camarilla_midpoint = (camarilla_r4_aligned[i] + camarilla_s4_aligned[i]) / 2
        
        long_exit = close[i] < camarilla_midpoint
        short_exit = close[i] > camarilla_midpoint
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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