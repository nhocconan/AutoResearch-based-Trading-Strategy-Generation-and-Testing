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
    
    # Load 1d data for price channel and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-period ATR on 1d
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.concatenate([[np.nan], high_1d[:-1]])),
            np.abs(low_1d - np.concatenate([[np.nan], low_1d[:-1]]))
        )
    )
    tr_1d[0] = np.nan
    
    atr_1d = np.full_like(high_1d, np.nan)
    for i in range(14, len(tr_1d)):
        if i == 14:
            atr_1d[i] = np.nanmean(tr_1d[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Calculate 1d high-low channel (20-period highest high and lowest low)
    highest_high_1d = np.full_like(high_1d, np.nan)
    lowest_low_1d = np.full_like(low_1d, np.nan)
    for i in range(19, len(high_1d)):
        highest_high_1d[i] = np.max(high_1d[i-19:i+1])
        lowest_low_1d[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 20-period volume average on 1d
    vol_avg_1d = np.full_like(volume_1d, np.nan)
    for i in range(19, len(volume_1d)):
        vol_avg_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align 1d indicators to 6h timeframe
    highest_high_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_high_1d)
    lowest_low_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after enough data for alignment
        # Skip if any critical data is NaN
        if (np.isnan(highest_high_1d_aligned[i]) or np.isnan(lowest_low_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period 1d average volume
        vol_ratio = volume[i] / vol_avg_1d_aligned[i] if vol_avg_1d_aligned[i] > 0 else 0
        
        if position == 0:
            # Long: breakout above 1d 20-period high with volume expansion
            if (close[i] > highest_high_1d_aligned[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: breakdown below 1d 20-period low with volume expansion
            elif (close[i] < lowest_low_1d_aligned[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint of 1d channel or volatility contraction
            midpoint = (highest_high_1d_aligned[i] + lowest_low_1d_aligned[i]) / 2
            if (close[i] < midpoint or
                vol_ratio < 0.8):  # Volume contraction
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midpoint of 1d channel or volatility contraction
            midpoint = (highest_high_1d_aligned[i] + lowest_low_1d_aligned[i]) / 2
            if (close[i] > midpoint or
                vol_ratio < 0.8):  # Volume contraction
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Volume_Expansion_Breakout_v1"
timeframe = "6h"
leverage = 1.0