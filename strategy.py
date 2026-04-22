#!/usr/bin/env python3
"""
Hypothesis: 6-hour Donchian breakout with weekly pivot direction and volume confirmation.
Long when price breaks above Donchian(20) high and weekly pivot direction is bullish with volume spike.
Short when price breaks below Donchian(20) low and weekly pivot direction is bearish with volume spike.
Exit when price returns to Donchian midpoint.
Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following weekly pivot trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Load weekly data for pivot calculation - ONCE before loop
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 30:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: P = (H+L+C)/3)
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    weekly_close = df_w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly pivot direction: bullish if current weekly close > previous weekly pivot
    weekly_pivot_dir = np.zeros_like(weekly_pivot)
    weekly_pivot_dir[1:] = (weekly_close[1:] > weekly_pivot[:-1]).astype(float) * 2 - 1  # 1 for bullish, -1 for bearish
    weekly_pivot_dir[0] = 0  # first bar undefined
    
    # Align weekly pivot direction to 6h timeframe
    weekly_pivot_dir_aligned = align_htf_to_ltf(prices, df_w, weekly_pivot_dir)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(weekly_pivot_dir_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above Donchian high, weekly pivot bullish, volume spike
            if (close[i] > donch_high[i] and 
                weekly_pivot_dir_aligned[i] > 0 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low, weekly pivot bearish, volume spike
            elif (close[i] < donch_low[i] and 
                  weekly_pivot_dir_aligned[i] < 0 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Donchian midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= Donchian midpoint
                if close[i] <= donch_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price >= Donchian midpoint
                if close[i] >= donch_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0