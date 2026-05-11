#!/usr/bin/env python3
name = "6h_Relative_Volume_Imbalance_Signal"
timeframe = "6h"
leverage = 1.0

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
    
    # 1d data for volume profile and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1-day average volume
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Calculate 6h volume ratio (current volume / 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma  # Current volume relative to recent average
    
    # Calculate price position within 6h range (0 = low, 1 = high)
    range_width = high - low
    range_width = np.where(range_width == 0, 1, range_width)  # Avoid division by zero
    price_position = (close - low) / range_width
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Ensure EMA50 and volume MA are ready
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(price_position[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: strong buying pressure (high close in range) + high volume + above daily EMA50
            if (price_position[i] > 0.7 and 
                vol_ratio[i] > 2.0 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong selling pressure (low close in range) + high volume + below daily EMA50
            elif (price_position[i] < 0.3 and 
                  vol_ratio[i] > 2.0 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: loss of buying pressure or volume drops
            if (price_position[i] < 0.4 or 
                vol_ratio[i] < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: loss of selling pressure or volume drops
            if (price_position[i] > 0.6 or 
                vol_ratio[i] < 1.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals