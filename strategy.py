#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel (trend detection)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) for breakout detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper = np.full(len(high_12h), np.nan)
    lower = np.full(len(high_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper[i] = np.max(high_12h[i-20:i])
        lower[i] = np.min(low_12h[i-20:i])
    donch_upper_12h = upper
    donch_lower_12h = lower
    donch_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_upper_12h)
    donch_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_lower_12h)
    
    # Get 1d data for volume filter (volume confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20) for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian and volume data
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_12h_aligned[i]) or np.isnan(donch_lower_12h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_12h_aligned[i]
        lower = donch_lower_12h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 1d MA (volume breakout)
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: breakout with volume
        if position == 0:
            # Long: break above upper band + volume
            if close[i] > upper and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band + volume
            elif close[i] < lower and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below midpoint or volume drops
            midpoint = (upper + lower) / 2
            if close[i] < midpoint or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above midpoint or volume drops
            midpoint = (upper + lower) / 2
            if close[i] > midpoint or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_VolumeBreakout_12h1d_v2"
timeframe = "12h"
leverage = 1.0