#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: On 6h timeframe, use weekly pivot (from 1w) for trend direction, Donchian(20) breakout for entry timing, and volume confirmation (1.5x average) to filter false breakouts. Weekly pivot provides robust trend filter that works in both bull and bear markets by capturing higher timeframe structure. Donchian breakouts catch momentum moves, volume confirmation ensures participation. Target 12-30 trades/year on BTC/ETH/SOL.
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
    
    # Get 1d data for ATR and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate ATR(14) on 1d for stoploss and volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate volume MA(20) on 1d for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate weekly pivot from previous 1w bar
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    prev_close_1w = df_1w['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high_1w = np.where(np.isnan(prev_high_1w), df_1w['high'].values, prev_high_1w)
    prev_low_1w = np.where(np.isnan(prev_low_1w), df_1w['low'].values, prev_low_1w)
    prev_close_1w = np.where(np.isnan(prev_close_1w), df_1w['close'].values, prev_close_1w)
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    # Align weekly pivot to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate Donchian channels (20-period) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d ATR, 1d volume MA, Donchian(20)
    start_idx = max(14, 20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        
        # Trend filter: price relative to weekly pivot
        above_pivot = close_val > pivot_1w_aligned[i]
        below_pivot = close_val < pivot_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high_val > highest_high[i]  # New 20-period high
        breakout_down = low_val < lowest_low[i]   # New 20-period low
        
        # Volume confirmation: current volume > 1.5 * 1d average volume
        volume_confirmed = volume_val > (1.5 * vol_ma_1d_aligned[i])
        
        if position == 0:
            # Long: price above weekly pivot AND Donchian breakout up AND volume confirmed
            long_signal = above_pivot and breakout_up and volume_confirmed
            
            # Short: price below weekly pivot AND Donchian breakout down AND volume confirmed
            short_signal = below_pivot and breakout_down and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below weekly pivot OR ATR stoploss (2.5 * ATR)
            if (below_pivot) or (close_val < entry_price - 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly pivot OR ATR stoploss (2.5 * ATR)
            if (above_pivot) or (close_val > entry_price + 2.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0