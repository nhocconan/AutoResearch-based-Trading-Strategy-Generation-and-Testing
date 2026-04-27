#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-day high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = np.full(len(high_1d), np.nan)
    donch_low = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    # Align Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate daily average volume (20-day)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after Donchian and volume MA are ready
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_aligned[i]
        
        # Volume filter: require volume above daily average
        vol_filter = vol_now > vol_avg
        
        if position == 0:
            # Long: price breaks above daily Donchian high with volume confirmation
            if price > donch_high_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below daily Donchian low with volume confirmation
            elif price < donch_low_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below daily Donchian low
            if price < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above daily Donchian high
            if price > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_1dBreakout_Volume"
timeframe = "6h"
leverage = 1.0