#!/usr/bin/env python3
"""
6h_Supertrend_Trend_Follow
Hypothesis: Use Supertrend (ATR=10, mult=3) on daily timeframe as trend filter.
On 6h timeframe, enter long when price crosses above Supertrend line in uptrend,
and short when price crosses below Supertrend line in downtrend.
Exit when price crosses back over the Supertrend line.
Designed to capture trends in both bull and bear markets while avoiding whipsaws
through ATR-based dynamic support/resistance. Targets 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period=10):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def calculate_supertrend(high, low, close, atr_period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    atr = calculate_atr(high, low, close, atr_period)
    
    # Basic upper and lower bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final upper and lower bands
    final_ub = np.zeros_like(close)
    final_lb = np.zeros_like(close)
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if close[i-1] > final_ub[i-1]:
            direction[i] = 1
        elif close[i-1] < final_lb[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            final_ub[i] = min(basic_ub[i], final_ub[i-1])
            final_lb[i] = basic_lb[i]
        else:
            final_ub[i] = basic_ub[i]
            final_lb[i] = max(basic_lb[i], final_lb[i-1])
        
        if direction[i] == 1:
            supertrend[i] = final_lb[i]
        else:
            supertrend[i] = final_ub[i]
    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend on daily (ATR=10, multiplier=3)
    supertrend_1d, direction_1d = calculate_supertrend(high_1d, low_1d, close_1d, 10, 3)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(supertrend_1d_aligned[i]) or np.isnan(direction_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Uptrend: direction = 1
            if direction_1d_aligned[i] == 1:
                # Long: price crosses above Supertrend
                if price > supertrend_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Downtrend: direction = -1
            elif direction_1d_aligned[i] == -1:
                # Short: price crosses below Supertrend
                if price < supertrend_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses below Supertrend or trend changes
            if price < supertrend_1d_aligned[i] or direction_1d_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Supertrend or trend changes
            if price > supertrend_1d_aligned[i] or direction_1d_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Supertrend_Trend_Follow"
timeframe = "6h"
leverage = 1.0