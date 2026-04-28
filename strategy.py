#!/usr/bin/env python3
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
    
    # Get 12h data for 12h Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Donchian channel (20-period)
    high_max_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_min_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, high_max_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, low_min_12h)
    
    # Get 1d data for volume filter and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d volume MA (20-period)
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR (14-period) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 4h timeframe
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_12h_aligned[i]) or 
            np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(volume_ma_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume above 1d average
        vol_filter = volume[i] > volume_ma_1d_aligned[i]
        
        # Volatility filter: ATR above 1d average (avoid low volatility chop)
        vol_filter = vol_filter and (atr_1d_aligned[i] > 0)  # Ensure ATR is valid
        
        # Breakout conditions: price breaks 12h Donchian with volume filter
        long_breakout = close[i] > donchian_high_12h_aligned[i]
        short_breakout = close[i] < donchian_low_12h_aligned[i]
        
        long_entry = long_breakout and vol_filter
        short_entry = short_breakout and vol_filter
        
        # Exit conditions: price returns to opposite Donchian band
        long_exit = close[i] < donchian_low_12h_aligned[i]
        short_exit = close[i] > donchian_high_12h_aligned[i]
        
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hBreakout_VolumeFilter"
timeframe = "4h"
leverage = 1.0