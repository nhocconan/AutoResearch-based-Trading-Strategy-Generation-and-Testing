#!/usr/bin/env python3
"""
6h Donchian(20) + Weekly Pivot Direction + Volume Confirmation
Hypothesis: Donchian breakouts capture momentum, weekly pivot provides directional bias from higher timeframe,
and volume confirms institutional participation. Works in bull (buy breakouts above pivot) and bear
(sell breakdowns below pivot). Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_w = get_htf_data(prices, '1w')
    
    # Weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot = (high_w + low_w + close_w) / 3.0
    r1 = 2 * pivot - low_w
    s1 = 2 * pivot - high_w
    
    # Weekly trend: price above/below pivot
    weekly_bullish = close_w > pivot
    weekly_bearish = close_w < pivot
    
    # Align to 6s timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_w, weekly_bearish)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # For Donchian channels
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ema[i]) or np.isnan(weekly_bullish_aligned[i]) or 
            np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: opposite signal or stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR stoploss
            if (close[i] <= donchian_low[i] or 
                close[i] <= entry_price - 2.5 * (high[i] - low[i])):  # ATR proxy
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR stoploss
            if (close[i] >= donchian_high[i] or 
                close[i] >= entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly pivot direction + volume
            bull_breakout = close[i] > donchian_high[i-1]  # Break above previous high
            bear_breakout = close[i] < donchian_low[i-1]   # Break below previous low
            volume_ok = volume[i] > vol_ema[i] * 1.5
            
            if bull_breakout and weekly_bullish_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and weekly_bearish_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals