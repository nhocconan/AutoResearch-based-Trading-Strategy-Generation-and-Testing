#!/usr/bin/env python3
# 1d_Supertrend_1wTrend_VolumeFilter
# Hypothesis: Supertrend on daily timeframe for trend direction, with weekly Supertrend confirmation and volume filter.
# Uses Supertrend(ATR=10, multiplier=3) on 1d and 1w timeframes. Requires both timeframes to agree on trend direction
# and volume above 20-period average to enter. Exits when trend direction changes on either timeframe.
# Designed to capture major trends while avoiding whipsaws in ranging markets. Targets 15-25 trades/year.

name = "1d_Supertrend_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    atr = np.zeros_like(close)
    atr[:period] = np.nan
    for i in range(period, len(close)):
        atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
    return atr

def supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[:period] = np.nan
    direction[:period] = np.nan
    
    for i in range(period, len(close)):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
            
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    supertrend_1d, direction_1d = supertrend(high_1d, low_1d, close_1d, period=10, multiplier=3)
    
    # Get 1w data for Supertrend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on 1w
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    supertrend_1w, direction_1w = supertrend(high_1w, low_1w, close_1w, period=10, multiplier=3)
    
    # Align 1d Supertrend direction to 1d timeframe (no alignment needed as we're on 1d)
    direction_1d_aligned = direction_1d
    
    # Align 1w Supertrend direction to 1d timeframe
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w.astype(float))
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(direction_1d_aligned[i]) or np.isnan(direction_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 2.0
        
        if position == 0:
            # Long: both timeframes show uptrend and volume filter passes
            if (direction_1d_aligned[i] > 0 and 
                direction_1w_aligned[i] > 0 and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: both timeframes show downtrend and volume filter passes
            elif (direction_1d_aligned[i] < 0 and 
                  direction_1w_aligned[i] < 0 and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: either timeframe shows downtrend
            if (direction_1d_aligned[i] < 0 or 
                direction_1w_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: either timeframe shows uptrend
            if (direction_1d_aligned[i] > 0 or 
                direction_1w_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals