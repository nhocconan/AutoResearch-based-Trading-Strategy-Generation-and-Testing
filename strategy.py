#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation.
Long when price breaks above Donchian high(20) and weekly pivot shows bullish bias (price > weekly pivot).
Short when price breaks below Donchian low(20) and weekly pivot shows bearish bias (price < weekly pivot).
Exit when price crosses back below Donchian median (long) or above Donchian median (short).
Weekly pivot calculated from weekly high/low/close: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H.
Volume filter: current volume > 1.5x 20-period average to avoid false breakouts.
Designed to generate 15-30 trades/year per symbol with strong breakout edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def donchian_channels(high, low, period):
    """Calculate Donchian channels: upper, lower, middle"""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, middle

def calculate_weekly_pivot(high, low, close):
    """Calculate weekly pivot points: P, R1, S1"""
    n = len(high)
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    for i in range(n):
        pivot[i] = (high[i] + low[i] + close[i]) / 3.0
        r1[i] = 2 * pivot[i] - low[i]
        s1[i] = 2 * pivot[i] - high[i]
    
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_high, donchian_low, donchian_middle = donchian_channels(high, low, 20)
    
    # Calculate weekly pivot points
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_pivot, weekly_r1, weekly_s1 = calculate_weekly_pivot(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    
    # Volume filter: volume > 1.5x average (to avoid false signals)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + volume MA (20)
    start_idx = max(20, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current values
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_middle[i]
        wp = weekly_pivot_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > weekly pivot + volume filter
            if price_now > d_high and price_now > wp and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low AND price < weekly pivot + volume filter
            elif price_now < d_low and price_now < wp and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below Donchian middle
            if price_now < d_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above Donchian middle
            if price_now > d_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0