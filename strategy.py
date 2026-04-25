#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with Weekly Pivot Direction and Volume Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum. Weekly pivot (from 1w data) provides 
longer-term bias: only take longs above weekly pivot, shorts below. Volume confirmation (>2.0x 20-bar 
vol MA) ensures breakout validity. Works in bull markets via upside breakouts above weekly pivot and 
in bear markets via downside breakdowns below weekly pivot. Discrete sizing (0.25) limits fee drag. 
Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly pivot calculation (need daily to build weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least a week of daily data
        return np.zeros(n)
    
    # Calculate weekly pivot from daily OHLC (using prior week's data)
    # Group daily data into weeks (starting Monday) and get prior week's OHLC
    # We'll approximate by using rolling weekly window on daily data
    # Weekly high = max of prior 5 daily highs, weekly low = min of prior 5 daily lows, weekly close = prior 5th daily close
    if len(df_1d) >= 5:
        # Calculate weekly OHLC from daily (prior week completed)
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values  # Prior week
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
        
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
        
        # Align weekly pivot to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # Calculate Donchian(20) channels on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values  # Prior 20 bars
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, weekly pivot, and volume MA
    start_idx = max(20, 20)  # 20 for Donchian (20 + 1 for shift), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above Donchian high AND above weekly pivot AND volume confirmation
            long_signal = (curr_high > dc_high) and (curr_close > weekly_pivot_val) and volume_confirm
            # Short: break below Donchian low AND below weekly pivot AND volume confirmation
            short_signal = (curr_low < dc_low) and (curr_close < weekly_pivot_val) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes back below Donchian high OR below weekly pivot
            if (curr_close < dc_high) or (curr_close < weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes back above Donchian low OR above weekly pivot
            if (curr_close > dc_low) or (curr_close > weekly_pivot_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirm"
timeframe = "6h"
leverage = 1.0