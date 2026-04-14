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
    
    # Load daily data (HTF) - ONCE BEFORE LOOP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from daily data
    def calculate_camarilla(high_val, low_val, close_val):
        pivot = (high_val + low_val + close_val) / 3
        range_val = high_val - low_val
        r3 = pivot + range_val * 1.1 / 2
        s3 = pivot - range_val * 1.1 / 2
        r4 = pivot + range_val * 1.1
        s4 = pivot - range_val * 1.1
        return pivot, r3, s3, r4, s4
    
    # Calculate Camarilla levels for each daily bar
    pivots = np.full(len(close_1d), np.nan)
    r3_levels = np.full(len(close_1d), np.nan)
    s3_levels = np.full(len(close_1d), np.nan)
    r4_levels = np.full(len(close_1d), np.nan)
    s4_levels = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        pivot, r3, s3, r4, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pivots[i] = pivot
        r3_levels[i] = r3
        s3_levels[i] = s3
        r4_levels[i] = r4
        s4_levels[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    pivots_aligned = align_htf_to_ltf(prices, df_1d, pivots)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_levels)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_levels)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_levels)
    
    # Calculate 20-period EMA for 6h trend
    if len(close) < 20:
        return np.zeros(n)
    
    ema20 = np.full_like(close, np.nan)
    alpha = 2 / (20 + 1)
    ema20[0] = close[0]
    for i in range(1, len(close)):
        ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    # Calculate 20-period volume average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size: 25% of capital
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivots_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(ema20[i]) or
            np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current period volume vs 20-period average
        volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above R4 with volume surge
            if (close[i] > r4_aligned[i] and 
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S4 with volume surge
            elif (close[i] < s4_aligned[i] and 
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below pivot
            if close[i] < pivots_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above pivot
            if close[i] > pivots_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0