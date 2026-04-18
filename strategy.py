#!/usr/bin/env python3
"""
12h_TurtleTrader_20_40_v1
12h strategy using dual Donchian channels with ATR-based exits.
- Long: Price breaks above 20-period Donchian high + price > 40-period SMA
- Short: Price breaks below 20-period Donchian low + price < 40-period SMA
- Exit: Price crosses 20-period Donchian opposite channel
- Filter: Volume > 1.5x 20-period average for breakout confirmation
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Uses proven Turtle Trading principles adapted for crypto markets
Works in trending markets (breakouts) and avoids ranging markets via volume filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 40-period SMA for trend filter
    sma_40 = pd.Series(close).rolling(window=40, min_periods=40).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 40 for SMA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(sma_40[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: break above Donchian high + price > SMA40 + volume
            if high[i] > high_20[i] and close[i] > sma_40[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + price < SMA40 + volume
            elif low[i] < low_20[i] and close[i] < sma_40[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low
            if low[i] < low_20[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high
            if high[i] > high_20[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TurtleTrader_20_40_v1"
timeframe = "12h"
leverage = 1.0