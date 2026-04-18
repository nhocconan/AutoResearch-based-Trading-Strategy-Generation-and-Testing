#!/usr/bin/env python3
"""
6h_12h_Multiplier_Volume_Filtered_Breakout
Hypothesis: Use 12h price action relative to 12h EMA34 as directional bias, with 6h Donchian breakout and volume confirmation.
Long when 6h price breaks above Donchian(20) high, 12h price > 12h EMA34, and volume > 2x 20-period average.
Short when 6h price breaks below Donchian(20) low, 12h price < 12h EMA34, and volume > 2x 20-period average.
Position size 0.25. Target: 20-40 trades/year per symbol (80-160 total over 4 years) to avoid fee drag.
Works in bull/bear via EMA trend filter and volume surge requirement.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA34 for trend direction
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 6h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge filter
        vol_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + 12h above EMA34 + volume surge
            if (close[i] > high_20[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                vol_surge):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + 12h below EMA34 + volume surge
            elif (close[i] < low_20[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_surge):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or 12h crosses below EMA34
            if (close[i] < low_20[i] or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or 12h crosses above EMA34
            if (close[i] > high_20[i] or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_Multiplier_Volume_Filtered_Breakout"
timeframe = "6h"
leverage = 1.0