#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d EMA Filter and Volume Confirmation v1
Hypothesis: Donchian(20) breakouts capture strong trends; 1d EMA50 filters counter-trend trades; volume confirms breakout strength. Works in bull (breakouts continue) and bear (breaks below lower band with trend filter). Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 20)  # Donchian and volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian break EMA or volume drop
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR EMA50
            if close[i] < lowest_low[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR EMA50
            if close[i] > highest_high[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + EMA filter + volume
            bull_break = close[i] > highest_high[i]
            bear_break = close[i] < lowest_low[i]
            vol_filter = volume[i] > vol_ma[i] * 1.5
            
            # Long: break above upper band with price above EMA50 (uptrend)
            if bull_break and close[i] > ema_50_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with price below EMA50 (downtrend)
            elif bear_break and close[i] < ema_50_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals