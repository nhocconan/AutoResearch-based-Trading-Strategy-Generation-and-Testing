#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivotDirection_Volume
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (from weekly pivot high/low) and volume confirmation. 
Weekly pivot levels provide structural support/resistance, Donchian breakouts capture momentum, 
and volume confirms breakout strength. Designed for medium frequency (50-150 total trades over 4 years) 
with performance in both bull and bear markets by filtering breakouts with weekly trend and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = np.full(len(close_1w), np.nan)
    r1 = np.full(len(close_1w), np.nan)
    s1 = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if i == 0:  # First week has no previous week
            continue
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        pivot[i] = (ph + pl + pc) / 3.0
        r1[i] = 2 * pivot[i] - pl
        s1[i] = 2 * pivot[i] - ph
    
    # Weekly trend: price above/below pivot
    weekly_bullish = close_1w > pivot
    weekly_bearish = close_1w < pivot
    
    # Align weekly data to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with weekly bullish bias and volume spike
            if (close[i] > donchian_high[i] and weekly_bullish_aligned[i] == 1.0 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with weekly bearish bias and volume spike
            elif (close[i] < donchian_low[i] and weekly_bearish_aligned[i] == 1.0 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian low or weekly bias turns bearish
            if (close[i] < donchian_low[i] or weekly_bearish_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or weekly bias turns bullish
            if (close[i] > donchian_high[i] or weekly_bullish_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivotDirection_Volume"
timeframe = "6h"
leverage = 1.0