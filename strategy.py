#!/usr/bin/env python3
"""
4h Camarilla Pivot R1/S1 Breakout with 1d Supertrend Filter and Volume Spike.
Long when: 1) Price breaks above R1, 2) 1d Supertrend = bullish, 3) Volume > 1.5x 20-period average.
Short when: 1) Price breaks below S1, 2) 1d Supertrend = bearish, 3) Volume > 1.5x 20-period average.
Exit when price returns to pivot point (mean reversion).
Designed for 4h timeframe: targets 75-200 total trades over 4 years (19-50/year).
Uses 1d Supertrend (ATR=10, mult=3) as trend filter to avoid whipsaws.
"""

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
    
    # Get 1d data for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend on daily timeframe
    atr_period = 10
    multiplier = 3
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR
    atr = np.full_like(close_1d, np.nan, dtype=np.float64)
    for i in range(atr_period, len(close_1d)):
        if i == atr_period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full_like(close_1d, np.nan, dtype=np.float64)
    direction = np.full_like(close_1d, np.nan, dtype=np.float64)  # 1 for up, -1 for down
    
    for i in range(atr_period, len(close_1d)):
        if i == atr_period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == upper_band[i-1]:
                if close_1d[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
            else:
                if close_1d[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
    
    # Supertrend direction (1 = bullish, -1 = bearish)
    supertrend_dir = direction
    
    # Align Supertrend direction to 4h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, supertrend_dir)
    
    # Calculate pivot points and Camarilla levels from previous day
    # We need daily high, low, close to calculate today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align daily levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Supertrend (10 periods), daily data, volume MA (20 periods)
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        trend = supertrend_dir_aligned[i]
        pivot_level = pivot_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above R1 + bullish trend + volume spike
            if price > r1_level and trend > 0 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 + bearish trend + volume spike
            elif price < s1_level and trend < 0 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot (mean reversion)
            if price <= pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot (mean reversion)
            if price >= pivot_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dSupertrend_Volume"
timeframe = "4h"
leverage = 1.0