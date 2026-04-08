#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_breakout_volume_v1"
timeframe = "12h"
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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Vectorized Donchian calculation with min_periods
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_12h, high_20)
    lower_12h = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Get 1d data for volume baseline
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume confirmation: 12h volume > 1.5x 1d average volume (scaled to 12h)
    # 1d volume represents 2x 12h periods, so divide by 2 for per-period comparison
    vol_threshold = vol_avg_1d_aligned / 2.0
    vol_confirm = volume > vol_threshold * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if Donchian or volume average not available
        if np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian lower band
            if close[i] < lower_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian upper band
            if close[i] > upper_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 12h Donchian upper band with volume confirmation
            if close[i] > upper_12h[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 12h Donchian lower band with volume confirmation
            elif close[i] < lower_12h[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals