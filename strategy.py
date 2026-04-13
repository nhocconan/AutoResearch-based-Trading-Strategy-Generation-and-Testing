#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day pivot direction and volume confirmation.
# Long: price breaks above Donchian(20) high + price above daily pivot + volume > 1.3x average volume
# Short: price breaks below Donchian(20) low + price below daily pivot + volume > 1.3x average volume
# Daily pivot from 1d data: PP = (high+low+close)/3
# Exit: price breaks back below/above Donchian opposite band
# Volume confirmation reduces false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe
# Works in both bull and bear markets by using daily pivot as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for daily pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point (using previous day's data)
    pp = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
    
    # Align 1d pivot to 4h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Donchian(20) on 4h timeframe
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        pivot = pp_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: break above Donchian high + above pivot + volume confirmation
            if (price > donch_high[i] and 
                price > pivot and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + below pivot + volume confirmation
            elif (price < donch_low[i] and 
                  price < pivot and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if price < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if price > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Pivot_Volume"
timeframe = "4h"
leverage = 1.0