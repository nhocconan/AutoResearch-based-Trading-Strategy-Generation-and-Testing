#!/usr/bin/env python3
"""
1h_4h_Donchian_Breakout_Volume_Filter_v1
Breakout strategy using 4h Donchian channels for direction and 1h volume confirmation for entry.
Designed to work in both bull and bear markets by capturing breakouts with volume confirmation.
Target: 60-150 total trades over 4 years (15-37/year).
Uses 4h for signal direction, 1h for entry timing and volume confirmation.
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
    
    # === 4h Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian upper and lower bands
    donch_high = np.full_like(high_4h, np.nan)
    donch_low = np.full_like(low_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:  # 20-period lookback
            donch_high[i] = np.max(high_4h[i-19:i+1])
            donch_low[i] = np.min(low_4h[i-19:i+1])
        else:
            donch_high[i] = np.nan
            donch_low[i] = np.nan
    
    # Align 4h Donchian bands to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === 1h Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:  # 20-period
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 20
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 4h Donchian high AND volume confirmation
            if (close[i] > donch_high_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: price breaks below 4h Donchian low AND volume confirmation
            elif (close[i] < donch_low_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_Donchian_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0