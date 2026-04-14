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
    
    # Load 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly pivot points (using prior week's OHLC)
    pivot_point = np.full_like(close_1w, np.nan)
    resistance1 = np.full_like(close_1w, np.nan)
    support1 = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= 2:
        for i in range(1, len(close_1w)):
            ph = high_1w[i-1]
            pl = low_1w[i-1]
            pc = close_1w[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            s1 = 2 * pp - ph
            
            pivot_point[i] = pp
            resistance1[i] = r1
            support1[i] = s1
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume SMA20
    vol_sma_20_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_sma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align weekly pivot points to 6h timeframe
    pivot_point_6h = align_htf_to_ltf(prices, df_1w, pivot_point)
    resistance1_6h = align_htf_to_ltf(prices, df_1w, resistance1)
    support1_6h = align_htf_to_ltf(prices, df_1w, support1)
    
    # Align daily volume SMA20 to 6h timeframe
    vol_sma_20_1d_6h = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_point_6h[i]) or 
            np.isnan(resistance1_6h[i]) or 
            np.isnan(support1_6h[i]) or
            np.isnan(vol_sma_20_1d_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs daily volume SMA20
        if vol_sma_20_1d_6h[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_sma_20_1d_6h[i]
        
        if position == 0:
            # Long: Price crosses above S1 with volume spike and above weekly pivot
            if (close[i] > support1_6h[i] and
                close[i] > pivot_point_6h[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below R1 with volume spike and below weekly pivot
            elif (close[i] < resistance1_6h[i] and
                  close[i] < pivot_point_6h[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below weekly pivot
            if close[i] < pivot_point_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above weekly pivot
            if close[i] > pivot_point_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_S1R1_Volume_Filter"
timeframe = "6h"
leverage = 1.0