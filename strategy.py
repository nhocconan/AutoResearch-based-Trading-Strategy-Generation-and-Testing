#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter and volume spike confirmation.
The weekly timeframe provides a strong trend filter that works in both bull and bear markets.
Volume spike (>2x 24-period average) confirms breakout strength. Designed for low trade frequency (7-25/year)
to minimize fee drag. Uses 1d timeframe for execution with weekly trend filter.
"""

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian(20) levels for trend filter
    donchian_high_1w = np.full_like(high_1w, np.nan)
    donchian_low_1w = np.full_like(low_1w, np.nan)
    if len(high_1w) >= 20:
        for i in range(19, len(high_1w)):
            donchian_high_1w[i] = np.max(high_1w[i-19:i+1])
            donchian_low_1w[i] = np.min(low_1w[i-19:i+1])
    
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Calculate daily Donchian(20) breakout levels
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    if len(high) >= 20:
        for i in range(19, len(high)):
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume spike filter: current volume / 24-period average volume (24*1d = 24 days)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        for i in range(23, len(volume)):
            vol_ma[i] = np.mean(volume[i-23:i+1])
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(23, 19)  # Ensure volume MA and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above daily Donchian high AND weekly uptrend (price > weekly Donchian high) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > donchian_high_1w_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below daily Donchian low AND weekly downtrend (price < weekly Donchian low) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < donchian_low_1w_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below daily Donchian low OR weekly trend reversal (price < weekly Donchian low)
                if close[i] < donchian_low[i] or close[i] < donchian_low_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 days
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above daily Donchian high OR weekly trend reversal (price > weekly Donchian high)
                if close[i] > donchian_high[i] or close[i] > donchian_high_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals