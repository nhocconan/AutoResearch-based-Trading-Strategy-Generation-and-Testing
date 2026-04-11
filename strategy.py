#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_v1"
timeframe = "1d"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high over last 20 weeks
    upper = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over last 20 weeks
    lower = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w average volume (20-period)
    volume_1w = df_1w['volume'].values
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1w indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    vol_avg_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_avg_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1w volume (aligned)
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        vol_surge = vol_1w_current > 1.5 * vol_avg_20_1w_aligned[i]  # 50% above average
        
        # Long when price breaks above upper Donchian band with volume surge
        long_signal = (close[i] > upper_aligned[i] and vol_surge)
        # Short when price breaks below lower Donchian band with volume surge
        short_signal = (close[i] < lower_aligned[i] and vol_surge)
        
        # Exit when price returns to the middle of the channel
        mid = (upper_aligned[i] + lower_aligned[i]) / 2
        exit_long = close[i] < mid
        exit_short = close[i] > mid
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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