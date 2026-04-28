#!/usr/bin/env python3
"""
4h_Donchian_Breakout_TopBottom_40
Hypothesis: Uses a 40-period Donchian channel breakout with price closing above/below the 20-period EMA for trend confirmation, and volume > 1.5x 20-period average volume to filter breakouts. Designed for low trade frequency (<30/year) to minimize fee burn while capturing strong directional moves in both bull and bear markets by requiring multi-condition alignment.
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
    
    # Calculate 40-period Donchian channel
    highest_high = pd.Series(high).rolling(window=40, min_periods=40).max().values
    lowest_low = pd.Series(low).rolling(window=40, min_periods=40).min().values
    
    # Calculate 20-period EMA for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Trend filter: price relative to EMA20
        uptrend = close[i] > ema_20[i]
        downtrend = close[i] < ema_20[i]
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Entry conditions
        long_entry = breakout_up and uptrend and vol_filter
        short_entry = breakout_down and downtrend and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = breakout_down or (close[i] < ema_20[i])
        short_exit = breakout_up or (close[i] > ema_20[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_TopBottom_40"
timeframe = "4h"
leverage = 1.0