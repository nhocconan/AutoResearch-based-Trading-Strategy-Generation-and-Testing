#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout strategy using 4h Donchian channels (20-period) and 1d volume confirmation.
# Breakouts from price channels capture momentum in trending markets while avoiding ranging whipsaws.
# Volume filter ensures breakouts have institutional participation.
# Uses 4h for signal direction (trend/channel) and 1h for precise entry timing.
# Session filter (08-20 UTC) reduces noise during low-volume hours.
# Position size fixed at 0.20 to control drawdown and enable multiple positions.
# Target: 60-150 total trades over 4 years (15-37/year) to stay within profitable range for 1h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period high/low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over last 20 periods
    upper_4h = np.full(len(high_4h), np.nan)
    # Lower band: lowest low over last 20 periods
    lower_4h = np.full(len(low_4h), np.nan)
    
    for i in range(20, len(high_4h)):
        upper_4h[i] = np.max(high_4h[i-20:i+1])
        lower_4h[i] = np.min(low_4h[i-20:i+1])
    
    # Calculate 1d average volume (20-period) for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i+1])
    
    # Current 1d volume (most recent)
    vol_current_1d = np.full(len(vol_1d), np.nan)
    for i in range(len(vol_1d)):
        vol_current_1d[i] = vol_1d[i]
    
    # Align 4h Donchian levels to 1h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Align 1d volume data to 1h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    vol_current_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_current_1d)
    
    # Pre-calculate session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # Fixed 20% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(vol_current_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours (08-20 UTC)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_4h_aligned[i]
        lower = lower_4h_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol_current = vol_current_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_filter = vol_current > 1.5 * vol_ma
        
        if position == 0:
            # Long breakout: price closes above upper Donchian band with volume confirmation
            if price > upper and volume_filter:
                position = 1
                signals[i] = position_size
            # Short breakdown: price closes below lower Donchian band with volume confirmation
            elif price < lower and volume_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower Donchian band (reversal signal)
            if price < lower:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper Donchian band (reversal signal)
            if price > upper:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Donchian_Volume_Breakout"
timeframe = "1h"
leverage = 1.0