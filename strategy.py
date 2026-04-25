#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
Hypothesis: Donchian channel breakouts capture momentum, while weekly pivot direction (from 1d HTF) filters for institutional bias. Volume confirmation ensures participation. Works in bull (long on upper break with bullish weekly pivot) and bear (short on lower break with bearish weekly pivot). Target: 12-37 trades/year on 6h.
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
    
    # Get 1d data for weekly pivot and Donchian (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points on 1d (using prior week's OHLC)
    # Approximate weekly pivot using rolling window on daily data
    # Weekly high = max(high over last 5 trading days)
    # Weekly low = min(low over last 5 trading days)
    # Weekly close = close of 5th day ago
    # Pivot = (weekly_high + weekly_low + weekly_close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close using 5-day lookback
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # prior week
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1).values
    
    # Weekly pivot
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bullish = weekly_close > weekly_pivot
    weekly_bearish = weekly_close < weekly_pivot
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    # Calculate Donchian channel (20-period) on 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and weekly pivot
    start_idx = max(20, 20)  # Donchian and volume MA both need 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_ma = vol_ma_20[i]
        pivot_val = weekly_pivot_aligned[i]
        is_bullish = weekly_bullish_aligned[i] > 0.5
        is_bearish = weekly_bearish_aligned[i] > 0.5
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high, weekly bullish, volume confirmation
            long_entry = (curr_high > donchian_high[i]) and is_bullish and volume_confirm
            # Short: price breaks below Donchian low, weekly bearish, volume confirmation
            short_entry = (curr_low < donchian_low[i]) and is_bearish and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below Donchian low OR weekly bias turns bearish
            if curr_low < donchian_low[i] or not is_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above Donchian high OR weekly bias turns bullish
            if curr_high > donchian_high[i] or not is_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dWeeklyPivot_VolumeConfirm"
timeframe = "6h"
leverage = 1.0