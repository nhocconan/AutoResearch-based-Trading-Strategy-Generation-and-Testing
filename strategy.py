#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation v1
Hypothesis: Donchian(20) breakouts capture strong trends. Weekly pivot levels filter direction (above weekly pivot = long bias, below = short bias). Volume confirms breakout strength. Works in bull (buy breakouts above weekly pivot) and bear (sell breakouts below weekly pivot). Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly pivot points (using previous week's OHLC)
    high_w = df_weekly['high'].values
    low_w = df_weekly['low'].values
    close_w = df_weekly['close'].values
    
    # Calculate pivot: P = (H + L + C) / 3
    pivot_w = (high_w + low_w + close_w) / 3.0
    # Support and resistance levels
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    r3_w = high_w + 2 * (pivot_w - low_w)
    s3_w = low_w - 2 * (high_w - pivot_w)
    
    # Align weekly data to 6h timeframe
    pivot_w_aligned = align_htf_to_ltf(prices, df_weekly, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_weekly, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_weekly, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_weekly, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_weekly, s2_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_weekly, r3_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_weekly, s3_w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and weekly data alignment
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(pivot_w_aligned[i]) or 
            np.isnan(r3_w_aligned[i]) or np.isnan(s3_w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout at S3/R3 or stoploss
        if position == 1:  # long position
            # Exit: price breaks below S3 OR stoploss
            if (close[i] <= s3_w_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R3 OR stoploss
            if (close[i] >= r3_w_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + pivot direction + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            # Long bias: price above weekly pivot
            bull_bias = close[i] > pivot_w_aligned[i]
            # Short bias: price below weekly pivot
            bear_bias = close[i] < pivot_w_aligned[i]
            
            # Volume confirmation
            vol_ok = volume[i] > vol_ema[i] * 1.5
            
            bull_entry = bull_breakout and bull_bias and vol_ok
            bear_entry = bear_breakout and bear_bias and vol_ok
            
            if bull_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals