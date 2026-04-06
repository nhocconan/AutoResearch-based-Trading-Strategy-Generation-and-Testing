#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 1d EMA Filter and Volume Confirmation
Hypothesis: Donchian breakouts capture trends; 1d EMA filters counter-trend moves;
volume confirms breakout strength. Works in bull (breakouts) and bear (filtered shorts).
Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for EMA filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start = max(20, 50)  # Donchian and EMA
    
    for i in range(start, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower OR below 1d EMA
            if close[i] < lowest_low_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper OR above 1d EMA
            if close[i] > highest_high_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + 1d EMA filter + volume
            bull_breakout = close[i] > highest_high_20[i]
            bear_breakout = close[i] < lowest_low_20[i]
            
            # 1d EMA filter: only long above EMA, short below EMA
            above_ema = close[i] > ema_50_1d_aligned[i]
            below_ema = close[i] < ema_50_1d_aligned[i]
            
            # Volume confirmation: above average
            vol_ok = volume[i] > vol_ma[i]
            
            if bull_breakout and above_ema and vol_ok:
                signals[i] = 0.25
                position = 1
            elif bear_breakout and below_ema and vol_ok:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals