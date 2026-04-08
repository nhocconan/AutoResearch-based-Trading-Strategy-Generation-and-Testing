#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v6"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume baseline
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 20-period average volume on 1d
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Volume confirmation: 4h volume > 1.5x daily average volume (scaled)
    # 1d volume represents 6x 4h periods, so divide by 6 for per-period comparison
    vol_threshold = vol_avg_20_1d_aligned / 6.0
    vol_confirm = volume > vol_threshold * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if Donchian or volume average not available
        if np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20-period Donchian low
            if close[i] < donchian_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 20-period Donchian high
            if close[i] > donchian_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period Donchian high with volume confirmation
            if close[i] > donchian_high_20_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period Donchian low with volume confirmation
            elif close[i] < donchian_low_20_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals