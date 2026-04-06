#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA Trend + Volume Filter
Hypothesis: On 4h timeframe, Donchian breakouts combined with 1d EMA trend filter and volume confirmation
capture significant moves while maintaining low trade frequency. 1d EMA filter ensures we only trade
in the direction of the higher timeframe trend, reducing whipsaws in both bull and bear markets.
Target: 75-200 total trades over 4 years (19-50/year) with low frequency to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1dema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 50-period EMA on 1d
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(20, 50)  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]):
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
            # Exit: price closes below Donchian lower OR price below 1d EMA
            if close[i] < lowest_low or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price above 1d EMA
            if close[i] > highest_high or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + 1d EMA trend filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            # Only go long if price above 1d EMA, short if below
            trend_uptrend = close[i] > ema_1d_aligned[i]
            trend_downtrend = close[i] < ema_1d_aligned[i]
            
            if i >= 20 and bull_breakout and volume_filter and trend_uptrend:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and trend_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals