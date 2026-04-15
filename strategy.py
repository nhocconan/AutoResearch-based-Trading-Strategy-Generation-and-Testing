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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate daily pivot points (standard floor trader's pivots)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Align HTF indicators to 4h timeframe with proper delay
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h Donchian channels (20-period) for breakout confirmation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_4h[i]) or np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above R1 with volume confirmation → long
        # 2. 4h price breaks below S1 with volume confirmation → short
        # 3. Additional confirmation: price must be above/below Donchian channel
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above R1 + above Donchian upper
        if (close[i] > r1_4h[i] and            # 4h price above R1 pivot
            close[i] > highest_20[i] and       # Above 20-period Donchian high
            volume_ratio[i] > 1.5):            # Strong volume confirmation
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below S1 + below Donchian lower
        elif (close[i] < s1_4h[i] and          # 4h price below S1 pivot
              close[i] < lowest_20[i] and      # Below 20-period Donchian low
              volume_ratio[i] > 1.5):          # Strong volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Pivot_R1_S1_Breakout_Donchian_Volume"
timeframe = "4h"
leverage = 1.0