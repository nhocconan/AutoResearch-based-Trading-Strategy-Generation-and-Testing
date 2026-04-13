#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Donchian breakout and volume confirmation.
# Long: Price breaks above Donchian(20) upper band + volume > 1.5x average volume.
# Short: Price breaks below Donchian(20) lower band + volume > 1.5x average volume.
# Exit: Price closes back inside the Donchian channel.
# Uses 1d Donchian channels for structure, 12h for execution with volume confirmation.
# Time filter: 00-23 UTC (all hours) to maximize opportunities while maintaining discipline.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donch_up = np.full(len(close_1d), np.nan)
    donch_low = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        donch_up[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Donchian levels to 12h
    donch_up_aligned = align_htf_to_ltf(prices, df_1d, donch_up)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_up_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donch_up_aligned[i]
        lower = donch_low_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above upper band + volume confirmation
            if (price > upper and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band + volume confirmation
            elif (price < lower and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes inside channel (below upper band)
            if price < upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes inside channel (above lower band)
            if price > lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0