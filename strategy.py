#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Confirmation
Hypothesis: Donchian channel breakouts on 12h with volume confirmation capture strong trends while avoiding false breakouts.
Long when price breaks above 20-period high + volume > 1.5x average.
Short when price breaks below 20-period low + volume > 1.5x average.
Exit on opposite breakout. Position size: ±0.25.
Designed to work in bull (captures rallies) and bear (captures declines) with volume filter reducing whipsaw.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous low
        
        if position == 0:
            # Long: upward breakout + volume filter
            if breakout_up and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume filter
            elif breakout_down and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: downward breakout
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: upward breakout
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0