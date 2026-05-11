#!/usr/bin/env python3
"""
1h_Supertrend_Filter_4hTrend_Direction
Hypothesis: Use 4h Supertrend for trend direction and 1h Supertrend for entry timing, with volume confirmation. 
Designed to capture trends in both bull and bear markets while avoiding whipsaws through multi-timeframe confirmation.
Limits trades by requiring alignment between 4h trend and 1h entry signal.
"""

name = "1h_Supertrend_Filter_4hTrend_Direction"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Average True Range
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Final Upper and Lower Bands
    final_ub = np.copy(basic_ub)
    final_lb = np.copy(basic_lb)
    
    # Supertrend
    supertrend = np.zeros_like(close)
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Determine Supertrend and direction
    for i in range(len(close)):
        if i == 0:
            supertrend[i] = final_lb[i]
            direction[i] = 1
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
                    direction[i] = -1
            else:
                if close[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
                    direction[i] = 1
    
    return supertrend, direction

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend for trend direction
    supertrend_4h, direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3)
    
    # Align 4h Supertrend direction to 1h timeframe
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h.astype(float))
    
    # Calculate 1h Supertrend for entry timing
    supertrend_1h, direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(direction_4h_aligned[i]) or 
            np.isnan(supertrend_1h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend AND 1h Supertrend buy signal AND volume filter
            if direction_4h_aligned[i] == 1 and direction_1h[i] == 1 and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend AND 1h Supertrend sell signal AND volume filter
            elif direction_4h_aligned[i] == -1 and direction_1h[i] == -1 and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h trend turns down OR 1h Supertrend sell signal
            if direction_4h_aligned[i] == -1 or direction_1h[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # maintain position
        elif position == -1:
            # Short exit: 4h trend turns up OR 1h Supertrend buy signal
            if direction_4h_aligned[i] == 1 or direction_1h[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # maintain position
    
    return signals