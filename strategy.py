#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data ONCE for HTF regime
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian channels (20-period)
    upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to 12h timeframe
    upper_20_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_20_1w)
    lower_20_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_20_1w)
    
    # Load daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # 12h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(upper_20_1w_aligned[i]) or np.isnan(lower_20_1w_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = upper_20_1w_aligned[i]
        lower = lower_20_1w_aligned[i]
        vol_ma = volume_ma_20_1d_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume filter: current volume must be above daily 20-period average
        vol_filter = vol > vol_ma
        
        if position == 0:
            # Long: price breaks above weekly upper Donchian with volume
            if price > upper and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower Donchian with volume
            elif price < lower and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly lower Donchian
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly upper Donchian
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyDonchian20_VolumeFilter"
timeframe = "12h"
leverage = 1.0