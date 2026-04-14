#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h trend filter and volume confirmation
# Uses 12h Donchian breakout (20-period) aligned to 6h chart with volume surge
# Works in bull/bear by capturing breakouts with institutional volume
# Target: 50-150 total trades over 4 years (12-37/year)
# Size: 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donch_high_12h = np.full_like(high_12h, np.nan)
    donch_low_12h = np.full_like(low_12h, np.nan)
    
    for i in range(19, len(high_12h)):
        donch_high_12h[i] = np.max(high_12h[i-19:i+1])
        donch_low_12h[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian levels to 6h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = np.full_like(volume_12h, np.nan)
    for i in range(19, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or 
            np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 12h-aligned volume vs its 20-period MA
        if vol_ma_12h_aligned[i] <= 0:
            volume_ratio = 0
        else:
            # Use the volume from the 12h bar that corresponds to this 6h bar
            # Get the volume of the most recent completed 12h bar
            vol_12h_current = volume_12h_aligned[i] if i < len(volume_12h_aligned) else volume_12h_aligned[-1]
            volume_ratio = vol_12h_current / vol_ma_12h_aligned[i] if vol_ma_12h_aligned[i] > 0 else 0
        
        if position == 0:
            # Long: price breaks above 12h Donchian high with volume surge
            if (close[i] > donch_high_12h_aligned[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 12h Donchian low with volume surge
            elif (close[i] < donch_low_12h_aligned[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low or volume drops
            if (close[i] < donch_low_12h_aligned[i] or
                volume_ratio < 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high or volume drops
            if (close[i] > donch_high_12h_aligned[i] or
                volume_ratio < 0.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0