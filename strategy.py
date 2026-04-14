#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe (wait for weekly bar close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 10-period Donchian channels on 1d
    upper_10 = np.full_like(high_1d, np.nan)
    lower_10 = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i < 9:
            upper_10[i] = np.nan
            lower_10[i] = np.nan
        else:
            upper_10[i] = np.max(high_1d[i-9:i+1])
            lower_10[i] = np.min(low_1d[i-9:i+1])
    
    # Align Donchian channels to 6h timeframe (wait for daily bar close)
    upper_10_aligned = align_htf_to_ltf(prices, df_1d, upper_10)
    lower_10_aligned = align_htf_to_ltf(prices, df_1d, lower_10)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # 20 for Donchian and volume
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_10_aligned[i]) or np.isnan(lower_10_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian (10) AND above weekly pivot with volume
            if price > upper_10_aligned[i] and price > pivot_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian (10) AND below weekly pivot with volume
            elif price < lower_10_aligned[i] and price < pivot_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below lower Donchian or below weekly pivot
            if price < lower_10_aligned[i] or price < pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above upper Donchian or above weekly pivot
            if price > upper_10_aligned[i] or price > pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_1d_Donchian_Pivot_Volume"
timeframe = "6h"
leverage = 1.0