#!/usr/bin/env python3
"""
12h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation
Hypothesis: Donchian breakouts capture momentum in trending markets. 
1d EMA200 filters for higher timeframe trend direction to avoid counter-trend trades.
Volume confirmation ensures institutional participation. Works in both bull (breakouts above EMA200) 
and bear (breakdowns below EMA200). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian20_1d_ema200_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_prev = np.roll(ema200_1d, 1)
    ema200_1d_prev[0] = ema200_1d[0]
    ema200_above = ema200_1d > ema200_1d_prev  # Uptrend
    ema200_below = ema200_1d < ema200_1d_prev  # Downtrend
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    ema200_above_aligned = align_htf_to_ltf(prices, df_1d, ema200_above)
    ema200_below_aligned = align_htf_to_ltf(prices, df_1d, ema200_below)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(200, lookback - 1)  # For EMA200 and Donchian
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(ema200_above_aligned[i]) or 
            np.isnan(ema200_below_aligned[i]) or np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR stoploss
            if (close[i] <= lowest_low[i] or 
                close[i] <= entry_price - 2.5 * (highest_high[i] - lowest_low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR stoploss
            if (close[i] >= highest_high[i] or 
                close[i] >= entry_price + 2.5 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            bull_entry = bull_breakout and ema200_above_aligned[i] and volume[i] > vol_ema[i] * 1.5
            bear_entry = bear_breakout and ema200_below_aligned[i] and volume[i] > vol_ema[i] * 1.5
            
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