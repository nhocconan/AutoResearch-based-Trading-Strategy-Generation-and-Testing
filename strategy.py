#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Donchian(20) breakouts on 12h capture strong trends. 1d EMA25 filters trend direction to avoid counter-trend trades. Volume confirms breakout strength. Works in bull (buy breakouts above) and bear (sell breakouts below). Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA25 for trend filter
    close_1d = df_1d['close'].values
    ema25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1d_prev = np.roll(ema25_1d, 1)
    ema25_1d_prev[0] = ema25_1d[0]
    ema25_rising = ema25_1d > ema25_1d_prev
    ema25_falling = ema25_1d < ema25_1d_prev
    ema25_1d_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    ema25_rising_aligned = align_htf_to_ltf(prices, df_1d, ema25_rising)
    ema25_falling_aligned = align_htf_to_ltf(prices, df_1d, ema25_falling)
    
    # 12h data
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
    start = 50  # For EMA25 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(ema25_1d_aligned[i]) or 
            np.isnan(ema25_rising_aligned[i]) or np.isnan(ema25_falling_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite breakout or stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian band OR stoploss
            if (close[i] <= lowest_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band OR stoploss
            if (close[i] >= highest_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            bull_entry = bull_breakout and ema25_rising_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and ema25_falling_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
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