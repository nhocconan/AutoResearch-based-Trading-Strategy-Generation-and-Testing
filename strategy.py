#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d EMA Filter and Volume Confirmation v1
Hypothesis: Donchian channel breakouts capture strong trends, while 1d EMA filter ensures
alignment with higher timeframe trend. Volume confirmation filters weak breakouts.
Designed for 75-200 trades over 4 years to minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # EMA on 1d close
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or EMA filter reversal
        if position == 1:  # long position
            # Exit: price closes below Donchian low OR price crosses below EMA
            if close[i] < lowest_low[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high OR price crosses above EMA
            if close[i] > highest_high[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + EMA filter + volume
            bull_breakout = close[i] > highest_high[i]
            bear_breakout = close[i] < lowest_low[i]
            
            ema_filter_long = close[i] > ema_1d_aligned[i]
            ema_filter_short = close[i] < ema_1d_aligned[i]
            
            volume_filter = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and ema_filter_long and volume_filter:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and ema_filter_short and volume_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals