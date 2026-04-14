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
    
    # Load 1d data for pivot points and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (S1, S2, R1, R2)
    pivot_point = np.full_like(close_1d, np.nan)
    resistance1 = np.full_like(close_1d, np.nan)
    resistance2 = np.full_like(close_1d, np.nan)
    support1 = np.full_like(close_1d, np.nan)
    support2 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            r2 = pp + (ph - pl)
            s1 = 2 * pp - ph
            s2 = pp - (ph - pl)
            
            pivot_point[i] = pp
            resistance1[i] = r1
            resistance2[i] = r2
            support1[i] = s1
            support2[i] = s2
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align 1d indicators to 6h timeframe
    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance1_6h = align_htf_to_ltf(prices, df_1d, resistance1)
    resistance2_6h = align_htf_to_ltf(prices, df_1d, resistance2)
    support1_6h = align_htf_to_ltf(prices, df_1d, support1)
    support2_6h = align_htf_to_ltf(prices, df_1d, support2)
    vol_ma_1d_6h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6h volume moving average (20-period)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_point_6h[i]) or 
            np.isnan(resistance1_6h[i]) or
            np.isnan(resistance2_6h[i]) or
            np.isnan(support1_6h[i]) or
            np.isnan(support2_6h[i]) or
            np.isnan(vol_ma_1d_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratios
        vol_ratio_6h = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        vol_ratio_1d = volume_1d[i//24] / vol_ma_1d_6h[i] if vol_ma_1d_6h[i] > 0 else 0
        
        # Volume thresholds
        vol_threshold_6h = 2.0
        vol_threshold_1d = 1.5
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike on both timeframes
            if (close[i] > resistance1_6h[i] and 
                vol_ratio_6h > vol_threshold_6h and
                vol_ratio_1d > vol_threshold_1d):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S1 with volume spike on both timeframes
            elif (close[i] < support1_6h[i] and 
                  vol_ratio_6h > vol_threshold_6h and
                  vol_ratio_1d > vol_threshold_1d):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below pivot point
            if close[i] < pivot_point_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above pivot point
            if close[i] > pivot_point_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Pivot_R1S1_Volume_BothTF"
timeframe = "6h"
leverage = 1.0