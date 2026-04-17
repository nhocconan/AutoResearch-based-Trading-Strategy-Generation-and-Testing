#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_v1
Breakout above/below weekly Donchian channel (20-period) with volume confirmation.
Exit when price returns to weekly midline (average of high/low over 20 weeks).
Designed to capture major trends with low frequency to minimize fee drag.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian Channel ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Align to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, mid_20)
    
    # === Volume confirmation (weekly average volume) ===
    vol_1w = df_1w['volume'].values
    vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x weekly average
        volume_ok = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly Donchian upper band with volume
            if (close[i] > donchian_upper_aligned[i] and volume_ok):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly Donchian lower band with volume
            elif (close[i] < donchian_lower_aligned[i] and volume_ok):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to weekly midline
        elif position == 1:
            # Exit long: price crosses below weekly midline
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly midline
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Donchian_Breakout_v1"
timeframe = "1d"
leverage = 1.0