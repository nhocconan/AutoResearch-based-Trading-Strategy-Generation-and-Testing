#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Donchian channel breakout and volume confirmation.
# Long: Price closes above Donchian upper (20) + volume > 1.5x average volume.
# Short: Price closes below Donchian lower (20) + volume > 1.5x average volume.
# Uses daily Donchian for structure, 4h for execution with volume confirmation.
# Target: 50-120 total trades over 4 years (12-30/year) for 4h timeframe.
# Designed to work in both bull and bear markets via breakout logic and volume filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channel (20-period) using previous day's data
    donchian_high = np.full(len(close), np.nan)
    donchian_low = np.full(len(close), np.nan)
    
    for i in range(20, len(high_1d)):
        # Use past 20 days (excluding current)
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Donchian levels to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price closes above upper Donchian + volume confirmation
            if (price > upper and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price closes below lower Donchian + volume confirmation
            elif (price < lower and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower Donchian (opposite side)
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper Donchian (opposite side)
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0