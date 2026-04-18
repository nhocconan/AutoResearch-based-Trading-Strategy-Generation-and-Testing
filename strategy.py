#!/usr/bin/env python3
"""
6h_WeeklyPivot_DonchianBreakout
Breakout strategy using weekly pivot levels for direction and daily Donchian breakouts for entry.
- Long when price breaks above daily Donchian(20) high AND weekly pivot shows bullish bias (price > weekly pivot)
- Short when price breaks below daily Donchian(20) low AND weekly pivot shows bearish bias (price < weekly pivot)
- Exit when price breaks opposite Donchian band
- Uses weekly pivot for trend filter to avoid counter-trend trades
- Designed for 15-25 trades/year per symbol
Works in both bull (captures uptrends) and bear (captures downtrends) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot point and support/resistance levels."""
    n = len(high)
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    
    for i in range(n):
        pivot[i] = (high[i] + low[i] + close[i]) / 3.0
        r1[i] = 2 * pivot[i] - low[i]
        s1[i] = 2 * pivot[i] - high[i]
        r2[i] = pivot[i] + (high[i] - low[i])
        s2[i] = pivot[i] - (high[i] - low[i])
        r3[i] = high[i] + 2 * (pivot[i] - low[i])
        s3[i] = low[i] - 2 * (high[i] - pivot[i])
    
    return pivot, r1, r2, r3, s1, s2, s3

def calculate_donchian(high, low, window=20):
    """Calculate Donchian channels."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(window-1, n):
        upper[i] = np.max(high[i-window+1:i+1])
        lower[i] = np.min(low[i-window+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot
    pivot_w, r1_w, r2_w, r3_w, s1_w, s2_w, s3_w = calculate_weekly_pivot(high_weekly, low_weekly, close_weekly)
    
    # Align weekly pivot to 6h timeframe
    pivot_w_6h = align_htf_to_ltf(prices, df_weekly, pivot_w)
    r1_w_6h = align_htf_to_ltf(prices, df_weekly, r1_w)
    r2_w_6h = align_htf_to_ltf(prices, df_weekly, r2_w)
    r3_w_6h = align_htf_to_ltf(prices, df_weekly, r3_w)
    s1_w_6h = align_htf_to_ltf(prices, df_weekly, s1_w)
    s2_w_6h = align_htf_to_ltf(prices, df_weekly, s2_w)
    s3_w_6h = align_htf_to_ltf(prices, df_weekly, s3_w)
    
    # Calculate daily Donchian for entry signals
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    donchian_upper_d, donchian_lower_d = calculate_donchian(high_daily, low_daily, window=20)
    
    # Align daily Donchian to 6h timeframe
    donchian_upper_6h = align_htf_to_ltf(prices, df_daily, donchian_upper_d)
    donchian_lower_6h = align_htf_to_ltf(prices, df_daily, donchian_lower_d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_w_6h[i]) or np.isnan(donchian_upper_6h[i]) or np.isnan(donchian_lower_6h[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly pivot bias
        bullish_bias = close[i] > pivot_w_6h[i]
        bearish_bias = close[i] < pivot_w_6h[i]
        
        # Check Donchian breakout on daily timeframe (aligned to 6h)
        breakout_upper = close[i] > donchian_upper_6h[i]
        breakout_lower = close[i] < donchian_lower_6h[i]
        
        if position == 0:
            # Long: bullish bias + break above daily Donchian upper
            if bullish_bias and breakout_upper:
                signals[i] = 0.25
                position = 1
            # Short: bearish bias + break below daily Donchian lower
            elif bearish_bias and breakout_lower:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below daily Donchian lower (reverse to short)
            if breakout_lower:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above daily Donchian upper (reverse to long)
            if breakout_upper:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_DonchianBreakout"
timeframe = "6h"
leverage = 1.0