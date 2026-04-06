#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h EMA Trend + Volume Confirmation
Hypothesis: Donchian breakouts on 6h capture medium-term momentum, filtered by 12h EMA trend to avoid counter-trend trades,
and volume confirmation ensures institutional participation. Designed for 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA(50) on 12h
    ema_50 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50[i] = close_12h[i] * 0.04 + ema_50[i-1] * 0.96  # alpha = 2/(50+1)
    
    # Align EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
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
        if np.isnan(ema_50_aligned[i]):
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
            # Exit: price closes below Donchian lower OR price < 12h EMA50
            if close[i] < lowest_low or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR price > 12h EMA50
            if close[i] > highest_high or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + 12h EMA trend
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif i >= 20 and bear_breakout and volume_filter and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals