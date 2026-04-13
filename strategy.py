#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Donchian breakout and volume confirmation.
# Long: Price breaks above 1d Donchian upper channel (20-period high) + volume > 1.5x average volume.
# Short: Price breaks below 1d Donchian lower channel (20-period low) + volume > 1.5x average volume.
# Uses 1d Donchian channels for trend structure, 12h for execution with volume confirmation.
# Exit when price crosses the middle of the Donchian channel (mean of upper/lower).
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
    
    # Calculate Donchian channels (20-period high/low) using previous day's data
    donchian_high = np.full(len(close_1d), np.nan)
    donchian_low = np.full(len(close_1d), np.nan)
    donchian_mid = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        hh = np.max(high_1d[i-20:i])
        ll = np.min(low_1d[i-20:i])
        donchian_high[i] = hh
        donchian_low[i] = ll
        donchian_mid[i] = (hh + ll) / 2.0
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Donchian levels to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        middle = donchian_mid_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above upper channel + volume confirmation
            if (price > upper and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower channel + volume confirmation
            elif (price < lower and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle channel
            if price < middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above middle channel
            if price > middle:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0