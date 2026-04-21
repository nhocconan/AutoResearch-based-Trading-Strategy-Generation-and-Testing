#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Daily Donchian(20) breakouts with volume confirmation (1.5x 20-day average) capture strong momentum moves in both bull and bear markets. Works because breakouts indicate institutional participation, and volume filters false breakouts. Targets 10-25 trades/year by requiring both price breakout and volume confirmation, reducing whipsaws. Uses tight stop (exit when price closes below Donchian low for longs, above high for shorts) to manage risk. Designed for 1d timeframe to minimize trade frequency and fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period high/low) - using daily data
    high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-period average volume
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    volume_ok = prices['volume'].values > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # Long: price breaks above 20-day high + volume confirmation
            if prices['high'].iloc[i] > high_20[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low + volume confirmation
            elif prices['low'].iloc[i] < low_20[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 20-day low
            if prices['close'].iloc[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 20-day high
            if prices['close'].iloc[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0