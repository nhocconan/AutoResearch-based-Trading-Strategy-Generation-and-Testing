#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily volume confirmation with 4h Donchian breakout.
# Enters long when price breaks above Donchian(20) high with above-average volume.
# Enters short when price breaks below Donchian(20) low with above-average volume.
# Uses daily volume filter to ensure institutional participation.
# Designed for 20-50 trades/year on 4h with clear risk management via signal reversal.
# Works in both bull and bear markets by capturing breakouts in direction of trend.

name = "4h_1d_donchian_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan, dtype=float)
    donchian_low = np.full_like(low, np.nan, dtype=float)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_avg_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * daily average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Breakout conditions
        breakout_long = high[i] >= donchian_high[i] and vol_filter
        breakout_low = low[i] <= donchian_low[i] and vol_filter
        
        # Exit conditions: reverse position on opposite breakout
        exit_long = position == 1 and breakout_low
        exit_short = position == -1 and breakout_long
        
        # Entry logic: breakout with volume confirmation
        if breakout_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakout_low and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = -1
            signals[i] = -0.25
        elif position == -1 and exit_short:
            position = 1
            signals[i] = 0.25
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals