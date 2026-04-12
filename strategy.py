#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_breakout_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period Donchian channels on weekly
    # Upper band: highest high over last 20 weeks
    donchian_upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 weeks
    donchian_lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 6h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume average on daily
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get current 6h volume for comparison
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current 6h volume > 1d 20-period average volume
        volume_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_upper_aligned[i] and volume_filter
        breakout_down = close[i] < donchian_lower_aligned[i] and volume_filter
        
        # Exit conditions: price crosses back through the opposite band
        exit_long = close[i] < donchian_lower_aligned[i]
        exit_short = close[i] > donchian_upper_aligned[i]
        
        # Execute trades
        if breakout_up and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals