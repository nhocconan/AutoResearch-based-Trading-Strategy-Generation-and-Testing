#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for calculations (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week Donchian channels
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 6h timeframe
    upper_donch_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    lower_donch_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Get daily data for volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-day average volume for spike detection
    vol_20d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = 20 + 5
    
    for i in range(start_idx, n):
        if (np.isnan(upper_donch_20w_aligned[i]) or 
            np.isnan(lower_donch_20w_aligned[i]) or 
            np.isnan(vol_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_20d_aligned[i] if vol_20d_aligned[i] > 0 else 0
        
        # Volume spike filter: at least 2x average daily volume
        vol_filter = vol_ratio > 2.0
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume
            if price > upper_donch_20w_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly Donchian low with volume
            elif price < lower_donch_20w_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price retests weekly Donchian low or volume drops
            if price < lower_donch_20w_aligned[i] or vol_ratio < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price retests weekly Donchian high or volume drops
            if price > upper_donch_20w_aligned[i] or vol_ratio < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyDonchian20_VolumeBreakout"
timeframe = "6h"
leverage = 1.0