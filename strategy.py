#!/usr/bin/env python3
"""
6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation.
Long when price breaks above Donchian upper band with weekly bullish bias and volume confirmation.
Short when price breaks below Donchian lower band with weekly bearish bias and volume confirmation.
Exit when price crosses back to middle band or weekly bias flips.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_direction_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOT DIRECTION (HTF) ===
    df_w = get_htf_data(prices, '1w')
    if len(df_w) == 0:
        return np.zeros(n)
    weekly_close = df_w['close'].values
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_prev = np.roll(weekly_pivot, 1)
    weekly_pivot_prev[0] = np.nan
    weekly_bullish = weekly_pivot > weekly_pivot_prev
    weekly_bearish = weekly_pivot < weekly_pivot_prev
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_w, weekly_bullish)
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_w, weekly_bearish)
    
    # === DONCHIAN CHANNEL (6H) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_band = (highest_high + lowest_low) / 2
    
    # === VOLUME CONFIRMATION (6H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below middle OR weekly bias turns bearish
            if close[i] < middle_band[i] or weekly_bearish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle OR weekly bias turns bullish
            if close[i] > middle_band[i] or weekly_bullish_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with weekly bias alignment
            if close[i] > highest_high[i] and weekly_bullish_aligned[i]:
                # Breakout above upper band in weekly bullish bias -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and weekly_bearish_aligned[i]:
                # Breakdown below lower band in weekly bearish bias -> short
                position = -1
                signals[i] = -0.25
    
    return signals