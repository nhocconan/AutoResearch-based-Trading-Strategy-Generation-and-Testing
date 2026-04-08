#!/usr/bin/env python3
"""
4H Donchian Breakout + Volume + Trend Filter
Hypothesis: Donchian(20) breakouts with volume confirmation and 4h EMA(50) trend filter capture strong momentum while avoiding whipsaws.
Designed for 4h timeframe to maintain low trade frequency (target 20-50/year) and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA(50) for trend filter on 4h
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(ema_50[i]) or np.isnan(vol_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses
            if close[i] < low_20[i] or close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses
            if close[i] > high_20[i] or close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long with trend and volume confirmation
            if (close[i] > high_20[i] and 
                close[i] > ema_50[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short with trend and volume confirmation
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals