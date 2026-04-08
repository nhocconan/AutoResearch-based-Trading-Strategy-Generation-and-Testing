#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume"
timeframe = "1d"
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
    
    # Get weekly data for Donchian channel and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channel on weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_channel = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (already delayed for completed weekly bars)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Volume threshold: 2x average volume
    vol_threshold = vol_avg_aligned * 2.0
    vol_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup period
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if Donchian or volume average not available
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(vol_avg_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below lower Donchian channel
            if close[i] < lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above upper Donchian channel
            if close[i] > upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with volume confirmation
            if close[i] > upper_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian with volume confirmation
            elif close[i] < lower_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals