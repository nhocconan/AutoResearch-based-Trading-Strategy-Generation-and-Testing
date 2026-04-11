#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1-day Donchian channel breakout and volume confirmation.
# Enters long when price breaks above 1-day upper Donchian (20-period) with volume > 1.5x 20-day average volume.
# Enters short when price breaks below 1-day lower Donchian with same volume filter.
# Exits when price returns to 1-day Donchian midpoint.
# Designed for 20-50 trades/year with strong trend capture and minimal whipsaw.

name = "4h_1d_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Upper and lower bands (20-period high/low)
    donch_high_20 = np.full_like(high_1d, np.nan, dtype=float)
    donch_low_20 = np.full_like(low_1d, np.nan, dtype=float)
    
    for i in range(19, len(high_1d)):
        donch_high_20[i] = np.max(high_1d[i-19:i+1])
        donch_low_20[i] = np.min(low_1d[i-19:i+1])
    
    # Donchian midpoint
    donch_mid_20 = (donch_high_20 + donch_low_20) / 2
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_20)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * daily average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Breakout conditions
        breakout_long = (high[i] >= donch_high_aligned[i]) and vol_filter
        breakout_short = (low[i] <= donch_low_aligned[i]) and vol_filter
        
        # Exit conditions: price returns to midpoint
        exit_long = (position == 1 and low[i] <= donch_mid_aligned[i])
        exit_short = (position == -1 and high[i] >= donch_mid_aligned[i])
        
        # Entry/exit logic
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals