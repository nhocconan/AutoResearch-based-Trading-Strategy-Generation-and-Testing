#!/usr/bin/env python3
"""
6h_Pivot_Fib618_Extension_Breakout_Volume
6h strategy using daily pivot points with Fibonacci 61.8% extension levels and volume confirmation.
- Long: Close breaks above daily pivot + 0.618 extension + volume > 1.5x daily avg
- Short: Close breaks below daily pivot - 0.618 extension + volume > 1.5x daily avg
- Exit: Opposite breakout or volume divergence
Designed for ~25-35 trades/year per symbol (100-140 total over 4 years)
Works in bull markets (breakout continuation) and bear markets (breakdown continuation)
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
    
    # Get daily data for pivot points and volume average
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot point and Fibonacci 61.8% extension
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Fibonacci 61.8% extension levels
    # Resistance = Pivot + 0.618 * Range
    # Support = Pivot - 0.618 * Range
    resistance_1d = pivot_1d + 0.618 * range_1d
    support_1d = pivot_1d - 0.618 * range_1d
    
    # Align daily levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance_1d)
    support_aligned = align_htf_to_ltf(prices, df_1d, support_1d)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(resistance_aligned[i]) or 
            np.isnan(support_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > resistance_aligned[i]
        breakdown_down = close[i] < support_aligned[i]
        
        if position == 0:
            # Long: volume + breakout above resistance (pivot + 0.618 ext)
            if vol_confirm and breakout_up:
                signals[i] = 0.25
                position = 1
            # Short: volume + breakdown below support (pivot - 0.618 ext)
            elif vol_confirm and breakdown_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: volume confirmation + breakdown below pivot
            if vol_confirm and close[i] < pivot_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: volume confirmation + breakout above pivot
            if vol_confirm and close[i] > pivot_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_Fib618_Extension_Breakout_Volume"
timeframe = "6h"
leverage = 1.0