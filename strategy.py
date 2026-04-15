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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly Supertrend (ATR=10, mult=3.0)
    # True Range
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (weekly_high + weekly_low) / 2.0
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(weekly_close)
    direction = np.ones_like(weekly_close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(weekly_close)):
        # Upper band logic
        if upper_band[i] < supertrend[i-1] or weekly_close[i-1] > supertrend[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = supertrend[i-1]
        
        # Lower band logic
        if lower_band[i] > supertrend[i-1] or weekly_close[i-1] < supertrend[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = supertrend[i-1]
        
        # Trend direction
        if direction[i-1] == -1 and weekly_close[i] > upper_band[i]:
            direction[i] = 1
        elif direction[i-1] == 1 and weekly_close[i] < lower_band[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Align weekly Supertrend direction to 6h timeframe
    supertrend_direction_6h = align_htf_to_ltf(prices, df_1w, direction)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_direction_6h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: Weekly uptrend + 6h price breaks above Donchian high + volume confirmation
        # Short: Weekly downtrend + 6h price breaks below Donchian low + volume confirmation
        # Discrete position sizing: 0.25
        
        # Long conditions
        if (supertrend_direction_6h[i] == 1 and            # Weekly uptrend
            close[i] > highest_20[i] and                   # 6h breakout above Donchian high
            volume_ratio[i] > 1.5):                        # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (supertrend_direction_6h[i] == -1 and         # Weekly downtrend
              close[i] < lowest_20[i] and                  # 6h breakdown below Donchian low
              volume_ratio[i] > 1.5):                      # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklySupertrend_Donchian_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0