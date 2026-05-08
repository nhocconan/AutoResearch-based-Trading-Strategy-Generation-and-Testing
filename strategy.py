#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Liquidity_Reversal"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for swing structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate swing points (local highs/lows)
    n1d = len(high_1d)
    swing_high = np.full(n1d, np.nan)
    swing_low = np.full(n1d, np.nan)
    
    for i in range(2, n1d-2):
        # Swing high: higher than 2 bars on each side
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = high_1d[i]
        # Swing low: lower than 2 bars on each side
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = low_1d[i]
    
    # Forward fill swing levels
    for i in range(1, n1d):
        if np.isnan(swing_high[i]):
            swing_high[i] = swing_high[i-1]
        if np.isnan(swing_low[i]):
            swing_low[i] = swing_low[i-1]
    
    # Align swing levels to 6h
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)
    
    # Volume imbalance: compare current volume to 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 1.0)
    
    # Price position relative to swing range
    range_width = swing_high_aligned - swing_low_aligned
    price_in_range = np.where(range_width > 0, (close - swing_low_aligned) / range_width, 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if critical data invalid
        if (np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near swing low with high volume (accumulation)
            long_cond = (price_in_range[i] < 0.2 and volume_ratio[i] > 1.8)
            # Short: price near swing high with high volume (distribution)
            short_cond = (price_in_range[i] > 0.8 and volume_ratio[i] > 1.8)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches middle of range or volume dries up
            if price_in_range[i] > 0.5 or volume_ratio[i] < 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches middle of range or volume dries up
            if price_in_range[i] < 0.5 or volume_ratio[i] < 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals