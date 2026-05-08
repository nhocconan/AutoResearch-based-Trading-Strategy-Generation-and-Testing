#!/usr/bin/env python3
"""
4h Donchian(20) breakout + volume confirmation + ATR filter
- Breakout above 20-period high with volume > 1.5x 20-period average
- Breakout below 20-period low with volume > 1.5x 20-period average
- ATR(14) filter: only trade when ATR > 0.5 * ATR(50) to avoid low volatility
- Position size: 0.25
- Designed for fewer trades (<50/year) to minimize fee drag on 4h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # ATR filter to avoid low volatility periods
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr14 > (0.5 * atr50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(atr_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high with volume and ATR filter
            long_cond = (close[i] > high_roll[i-1] and 
                        volume_filter[i] and
                        atr_filter[i])
            
            # Short: breakout below Donchian low with volume and ATR filter
            short_cond = (close[i] < low_roll[i-1] and 
                         volume_filter[i] and
                         atr_filter[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals