#!/usr/bin/env python3
"""
12h_WAVES_1W_Volume_Signal_v1
Hypothesis: Use weekly price waves (distance from weekly high/low) + volume confirmation.
In bull markets, buy near weekly lows; in bear markets, sell near weekly highs.
Volume filter ensures participation. Weekly timeframe reduces noise and overtrading.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly high/low for wave calculation ===
    df_1w = get_htf_data(prices, '1w')
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Align weekly data to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Calculate position within weekly range (0 = at low, 1 = at high)
    weekly_range = weekly_high_aligned - weekly_low_aligned
    # Avoid division by zero
    weekly_range = np.where(weekly_range == 0, 1e-10, weekly_range)
    wave_position = (close - weekly_low_aligned) / weekly_range
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(wave_position[i]) or 
            np.isnan(weekly_close_aligned[i]) or
            np.isnan(volume_ok[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: near weekly low (wave < 0.3) and volume confirmation
            if (wave_position[i] < 0.3 and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: near weekly high (wave > 0.7) and volume confirmation
            elif (wave_position[i] > 0.7 and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses weekly close OR wave > 0.8
            if (close[i] > weekly_close_aligned[i] or 
                wave_position[i] > 0.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses weekly close OR wave < 0.2
            if (close[i] < weekly_close_aligned[i] or 
                wave_position[i] < 0.2):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WAVES_1W_Volume_Signal_v1"
timeframe = "12h"
leverage = 1.0