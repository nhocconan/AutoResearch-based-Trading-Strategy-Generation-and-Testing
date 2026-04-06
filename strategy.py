#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA200 Filter + Volume Confirmation v1
Hypothesis: Donchian breakouts capture momentum; EMA200 on 1d filters counter-trend trades; volume confirms breakout strength.
Designed for 75-200 trades over 4 years to minimize fee drag while adapting to bull/bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA200 filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA200
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 4h
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(period, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout
        if position == 1:  # long position
            # Exit: price breaks below lower band
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper band
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA200 filter + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            # EMA200 filter: only long above EMA200, short below EMA200
            bull_filter = close[i] > ema200_1d_aligned[i]
            bear_filter = close[i] < ema200_1d_aligned[i]
            
            # Volume filter: volume above average
            vol_filter = volume[i] > vol_ma[i]
            
            if bull_breakout and bull_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and bear_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals