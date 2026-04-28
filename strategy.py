#!/usr/bin/env python3
# Hypothesis: 12h 20-period Donchian channel breakout with 1-week volume confirmation and 1-day ADX trend filter.
# In trending markets (ADX>25), breakouts of price channels tend to continue. Volume confirms participation.
# Works in both bull and bear markets by filtering for strong trends via ADX and requiring volume.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

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
    
    # Get weekly data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for volume average
        return np.zeros(n)
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate weekly average volume (20-period)
    weekly_volume = df_1w['volume'].values
    volume_ma = pd.Series(weekly_volume).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma)
    
    # Calculate daily ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def _smma(array, period):
        """Smoothed Moving Average (SMMA)"""
        if len(array) < period:
            return np.full_like(array, np.nan, dtype=float)
        result = np.full_like(array, np.nan, dtype=float)
        # First value is simple moving average
        result[period-1] = np.mean(array[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_close) / period
        for i in range(period, len(array)):
            result[i] = (result[i-1] * (period-1) + array[i]) / period
        return result
    
    atr = _smma(tr, 14)
    plus_di_smoothed = _smma(plus_dm, 14)
    minus_di_smoothed = _smma(minus_dm, 14)
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = _smma(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x weekly average volume
        volume_confirm = volume[i] > (volume_ma_aligned[i] * 1.5)
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i-1]  # Break above previous period's high
        short_breakout = close[i] < lowest_low[i-1]   # Break below previous period's low
        
        # Entry conditions
        long_entry = strong_trend and long_breakout and volume_confirm
        short_entry = strong_trend and short_breakout and volume_confirm
        
        # Exit conditions: when trend weakens or opposite breakout occurs
        long_exit = (not strong_trend) or (position == 1 and short_breakout)
        short_exit = (not strong_trend) or (position == -1 and long_breakout)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_1wVolume_1dADX_Trend"
timeframe = "12h"
leverage = 1.0