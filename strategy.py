#!/usr/bin/env python3
# 4H_Donchian20_Volume_Spike_Trend
# Hypothesis: Combines Donchian(20) breakout with volume spike and trend filter (100-period SMA).
# Designed for 4h timeframe with low trade frequency (<40/year) and strong performance in both bull and bear regimes.
# Long: Close > Donchian(20) high + Volume > 1.8x average volume + Close > SMA100.
# Short: Close < Donchian(20) low + Volume > 1.8x average volume + Close < SMA100.
# Exit: Opposite Donchian breakout or volume drop below average.
# Target: 20-50 trades per year per symbol.

name = "4H_Donchian20_Volume_Spike_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Trend filter: 100-period SMA
    sma100 = pd.Series(close).rolling(window=100, min_periods=100).mean().values
    
    # Volume filter: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for SMA100
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(sma100[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + Uptrend (close > SMA100) + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > sma100[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + Downtrend (close < SMA100) + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < sma100[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions:
            # 1. Opposite Donchian breakout
            # 2. Volume drops below average (loss of momentum)
            opposite_breakout = (position == 1 and close[i] < donchian_low[i]) or \
                               (position == -1 and close[i] > donchian_high[i])
            volume_drop = volume[i] < vol_ma[i]
            
            if opposite_breakout or volume_drop:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals