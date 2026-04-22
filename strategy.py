#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data once
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-week period)
    # Using previous week's data to avoid look-ahead
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    
    donchian_high_20w = pd.Series(prev_high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20w = pd.Series(prev_low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20w)
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6s data for entry timing
    volume_6h = prices['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_6h = volume_6h[i]
        vol_ma_6h_val = vol_ma_6h[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_1d_ma = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with 1d volume confirmation
            if price > upper and vol_6h > 1.5 * vol_ma_6h_val and volume_1d[i] > 1.5 * vol_1d_ma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with 1d volume confirmation
            elif price < lower and vol_6h > 1.5 * vol_ma_6h_val and volume_1d[i] > 1.5 * vol_1d_ma:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to the middle of the weekly Donchian channel
            mid = (upper + lower) / 2.0
            if position == 1 and price < mid:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WeeklyDonchian_Breakout_1dVolume_Confirmation"
timeframe = "6h"
leverage = 1.0