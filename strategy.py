#!/usr/bin/env python3
import numpy as np
import pandas as pd
from miflow.mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1w_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-period rolling max/min on weekly data
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 4h timeframe
    high_20_4h = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_4h = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20_4h[i]) or np.isnan(low_20_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > high_20_4h[i]
        breakout_down = close[i] < low_20_4h[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit when price breaks below weekly low or volume dries up
            if close[i] < low_20_4h[i] or volume[i] < vol_ma[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when price breaks above weekly high or volume dries up
            if close[i] > high_20_4h[i] or volume[i] < vol_ma[i] * 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: breakout above weekly high + volume confirmation
            if breakout_up and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: breakout below weekly low + volume confirmation
            elif breakout_down and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals