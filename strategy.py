#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1w_donchian_volume_tight_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-week Donchian channels
    high_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20w)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_ma_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ok_1d = volume_1d > vol_ma_20d
    
    # Align volume filter to 4h timeframe
    volume_ok = align_htf_to_ltf(prices, df_1d, volume_ok_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ok[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with weekly volume confirmation
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        vol_ok = volume_ok[i]
        
        # Enter on breakout with volume confirmation
        if breakout_up and vol_ok and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_down and vol_ok and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit on opposite breakout
        elif breakout_down and position == 1:
            position = 0
            signals[i] = 0.0
        elif breakout_up and position == -1:
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