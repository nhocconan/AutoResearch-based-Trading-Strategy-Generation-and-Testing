#!/usr/bin/env python3
"""
12h Donchian breakout with weekly trend filter and volume confirmation
Hypothesis: Donchian(20) breakouts capture institutional trends on 12h timeframe.
Weekly EMA50 filters trend direction to avoid counter-trend trades.
Volume confirms institutional participation. Works in both bull (breakout high) 
and bear (breakout low) markets. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_prev = np.roll(ema50_1w, 1)
    ema50_1w_prev[0] = ema50_1w[0]
    ema50_rising = ema50_1w > ema50_1w_prev
    ema50_falling = ema50_1w < ema50_1w_prev
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_falling)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR stoploss
            if (close[i] <= lowest_low[i] or 
                close[i] <= entry_price - 2.5 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR stoploss
            if (close[i] >= highest_high[i] or 
                close[i] >= entry_price + 2.5 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            long_breakout = close[i] > highest_high[i]
            short_breakout = close[i] < lowest_low[i]
            
            long_entry = long_breakout and ema50_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            short_entry = short_breakout and ema50_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals