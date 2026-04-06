#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d EMA Filter and Volume Confirmation v1
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
The 1d EMA filter ensures we only trade in the direction of the higher timeframe trend, 
reducing counter-trend trades. Volume confirmation ensures breakouts have conviction. 
Designed for 75-200 trades over 4 years to minimize fee drag while adapting to bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50, 20)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or close back inside channel
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR opposite breakout
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR opposite breakout
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + 1d EMA filter + volume
            bull_breakout = close[i] > donchian_upper[i]
            bear_breakout = close[i] < donchian_lower[i]
            
            # 1d EMA filter: only long if price > EMA, only short if price < EMA
            ema_filter_long = close[i] > ema_50_1d_aligned[i]
            ema_filter_short = close[i] < ema_50_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            volume_filter = volume[i] > vol_ma[i] * 1.5
            
            if bull_breakout and ema_filter_long and volume_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and ema_filter_short and volume_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals