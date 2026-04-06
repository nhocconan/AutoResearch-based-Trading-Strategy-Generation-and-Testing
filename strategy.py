#!/usr/bin/env python3
"""
6h Donchian(20) breakout with weekly pivot filter and volume confirmation
Hypothesis: Weekly pivot levels define major support/resistance. Donchian breakouts
in the direction of weekly trend (above/below weekly pivot) capture strong moves.
Volume > 1.5x 20-period EMA confirms breakout strength. Works in bull (buy above pivot)
and bear (sell below pivot). Target: 80-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly pivot points (using prior week's H/L/C)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support 1 = (2 * Pivot) - High
    weekly_s1 = (2 * weekly_pivot) - weekly_high
    # Resistance 1 = (2 * Pivot) - Low
    weekly_r1 = (2 * weekly_pivot) - weekly_low
    
    # Align weekly data to 6h timeframe (shifted by 1 for completed weeks)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    
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
    start = 50  # For Donchian and volume EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r1_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below weekly S1 OR stoploss
            if (close[i] <= weekly_s1_aligned[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above weekly R1 OR stoploss
            if (close[i] >= weekly_r1_aligned[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            # Only long above weekly pivot, only short below weekly pivot
            bull_entry = bull_breakout and (close[i] > weekly_pivot_aligned[i]) and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and (close[i] < weekly_pivot_aligned[i]) and volume[i] > vol_ema[i] * 1.5
            
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