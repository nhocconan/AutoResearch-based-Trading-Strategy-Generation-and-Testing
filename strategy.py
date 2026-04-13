#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation (main signal)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Calculate 20-period average volume on 12h
    vol_12h = df_12h['volume'].values
    avg_volume_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        avg_volume_12h[i] = np.mean(vol_12h[i-20:i])
    
    # Get 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR on 1d
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.full(len(high_1d), np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average volume
        vol_confirm = volume[i] > avg_volume_12h_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_long = close[i] > donchian_high_aligned[i]
        donchian_breakout_short = close[i] < donchian_low_aligned[i]
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[:i]) if i > 200 else True
        
        # Entry conditions with confluence
        long_entry = donchian_breakout_long and vol_confirm and vol_filter
        short_entry = donchian_breakout_short and vol_confirm and vol_filter
        
        # Exit conditions: opposite Donchian breakout
        exit_long = position == 1 and donchian_breakout_short
        exit_short = position == -1 and donchian_breakout_long
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_volume_volatility_filter"
timeframe = "12h"
leverage = 1.0