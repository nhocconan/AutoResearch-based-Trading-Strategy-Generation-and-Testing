#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly EMA Filter and Volume Confirmation v1
Hypothesis: Daily Donchian breakouts capture strong trends. Weekly EMA filter ensures
alignment with higher timeframe trend, reducing counter-trend trades. Volume
confirms breakout strength. Designed for 30-100 trades over 4 years to minimize
fee drag while adapting to bull/bear markets via weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Weekly EMA (20-period)
    ema_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 20)  # For Donchian and volume
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_weekly_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite Donchian breakout or weekly EMA flip
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian OR weekly EMA turns bearish
            if close[i] < lowest_low_20[i] or close[i] < ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian OR weekly EMA turns bullish
            if close[i] > highest_high_20[i] or close[i] > ema_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly EMA filter + volume
            bull_breakout = close[i] > highest_high_20[i]
            bear_breakout = close[i] < lowest_low_20[i]
            
            weekly_bias_up = close[i] > ema_weekly_aligned[i]
            weekly_bias_down = close[i] < ema_weekly_aligned[i]
            
            volume_filter = volume[i] > vol_ma[i] * 1.5
            
            bull_entry = bull_breakout and weekly_bias_up and volume_filter
            bear_entry = bear_breakout and weekly_bias_down and volume_filter
            
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