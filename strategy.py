#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d volume confirmation and session filter
    # Long: price breaks above 4h Donchian upper (20) AND 1d volume > 1.5x 20-day avg AND hour 8-20 UTC
    # Short: price breaks below 4h Donchian lower (20) AND 1d volume > 1.5x 20-day avg AND hour 8-20 UTC
    # Exit: price returns to 4h Donchian midpoint OR volume dry-up
    # Using 4h for structure (low trade frequency), 1d volume for confirmation,
    # session filter to avoid off-hours noise. Discrete position sizing (0.20).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_len = 20
    upper_4h = np.full(len(high_4h), np.nan)
    lower_4h = np.full(len(low_4h), np.nan)
    
    for i in range(donchian_len-1, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-donchian_len+1:i+1])
        lower_4h[i] = np.min(low_4h[i-donchian_len+1:i+1])
    
    # Align 4h Donchian to 1h
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    midpoint_4h = (upper_4h_aligned + lower_4h_aligned) / 2
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Volume spike condition: current volume > 1.5x 20-day average
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC (avoid off-hours noise)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session check
        if not session_filter[i]:
            # Outside session: flatten position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike_1d_aligned[i]
        
        # Entry logic: Donchian breakout + volume confirmation
        long_entry = (close[i] > upper_4h_aligned[i]) and vol_confirm
        short_entry = (close[i] < lower_4h_aligned[i]) and vol_confirm
        
        # Exit logic: return to midpoint OR volume dry-up
        long_exit = (close[i] < midpoint_4h[i]) or not vol_confirm
        short_exit = (close[i] > midpoint_4h[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_session_v1"
timeframe = "1h"
leverage = 1.0