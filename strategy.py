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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Calculate 20-period average volume on 1d
    avg_volume_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        avg_volume_1d[i] = np.mean(vol_1d[i-20:i])
    
    # Get 1h data for EMA filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    # Calculate 50-period EMA on 1h
    ema_50 = np.full(len(close_1h), np.nan)
    if len(close_1h) >= 50:
        ema_50[49] = np.mean(close_1h[:50])
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1h)):
            ema_50[i] = (close_1h[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    # Align all indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    ema_50_aligned = align_htf_to_ltf(prices, df_1h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average volume
        vol_confirm = volume[i] > avg_volume_1d_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_long = close[i] > donchian_high_aligned[i]
        donchian_breakout_short = close[i] < donchian_low_aligned[i]
        
        # EMA trend filter: price above EMA for long, below for short
        ema_filter_long = close[i] > ema_50_aligned[i]
        ema_filter_short = close[i] < ema_50_aligned[i]
        
        # Entry conditions with confluence
        long_entry = donchian_breakout_long and vol_confirm and ema_filter_long
        short_entry = donchian_breakout_short and vol_confirm and ema_filter_short
        
        # Exit conditions: opposite Donchian breakout or EMA reversal
        exit_long = position == 1 and (donchian_breakout_short or close[i] < ema_50_aligned[i])
        exit_short = position == -1 and (donchian_breakout_long or close[i] > ema_50_aligned[i])
        
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

name = "4h_1d_1h_donchian_ema_volume"
timeframe = "4h"
leverage = 1.0