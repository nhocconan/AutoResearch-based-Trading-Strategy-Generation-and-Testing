#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # Align 1d indicators to 1d timeframe (same timeframe)
    pivot_point_1d = pivot_point
    resistance1_1d = resistance1
    resistance2_1d = resistance2
    support1_1d = support1
    support2_1d = support2
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-period SMA on weekly
    sma_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        for i in range(49, len(close_1w)):
            sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    # Align 1w SMA to 1d timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume spike detection (20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_point_1d[i]) or 
            np.isnan(resistance1_1d[i]) or
            np.isnan(resistance2_1d[i]) or
            np.isnan(support1_1d[i]) or
            np.isnan(support2_1d[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.0
        
        # Trend filter: only trade in direction of weekly trend
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and in weekly uptrend
            if (close[i] > resistance1_1d[i] and 
                volume_ratio > vol_threshold and
                weekly_uptrend):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S1 with volume spike and in weekly downtrend
            elif (close[i] < support1_1d[i] and 
                  volume_ratio > vol_threshold and
                  not weekly_uptrend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below pivot point (mean reversion)
            if close[i] < pivot_point_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above pivot point (mean reversion)
            if close[i] > pivot_point_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Pivot_Volume_Trend_Filter"
timeframe = "1d"
leverage = 1.0