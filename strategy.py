#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolume_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_surge_1d = volume_1d > (volume_ma_1d * 1.5)  # 1d volume surge
    
    # Align 1d volume surge to 4h timeframe
    volume_surge_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_surge_1d)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Trend filter: price above/below 50-period EMA
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(ema_50[i]) or np.isnan(volume_surge_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        volume_surge = volume_surge_1d_aligned[i]
        
        if position == 0:
            # Long when price breaks above upper Donchian + volume surge + above EMA50
            if close[i] > high_max[i] and volume_surge and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian + volume surge + below EMA50
            elif close[i] < low_min[i] and volume_surge and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below lower Donchian
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above upper Donchian
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals