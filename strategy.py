#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w EMA Trend + Volume Filter
Hypothesis: On daily timeframe, Donchian breakouts combined with weekly trend filter and volume confirmation 
capture significant moves while maintaining low trade frequency. Weekly EMA filter ensures we only trade 
in the direction of the higher timeframe trend, reducing whipsaws in both bull and bear markets.
Target: 30-100 total trades over 4 years (7-25/year) with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 100-period EMA on weekly
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 100:
        multiplier = 2 / (100 + 1)
        ema_1w[0] = close_1w[0]
        for i in range(1, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 100)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR price below weekly EMA
            if close[i] < lowest_low or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price above weekly EMA
            if close[i] > highest_high or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly EMA trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Only go long if price above weekly EMA, short if below
            weekly_uptrend = close[i] > ema_1w_aligned[i]
            weekly_downtrend = close[i] < ema_1w_aligned[i]
            
            if i >= 20 and bull_breakout and volume_filter and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals