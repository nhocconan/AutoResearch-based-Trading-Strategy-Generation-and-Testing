#!/usr/bin/env python3
"""
4h_Donchian_20_Breakout_Volume_Trend_4hTrend
Uses 4h Donchian breakout with volume confirmation and 4h EMA trend filter.
Long: breakout above 20-period high + volume spike + price above EMA20.
Short: breakdown below 20-period low + volume spike + price below EMA20.
Exit: opposite Donchian break or close crossing EMA20.
Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drift.
Works in bull/bear markets by following 4h trend while using Donchian breakouts for entries.
"""

name = "4h_Donchian_20_Breakout_Volume_Trend_4hTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: breakout above high_20 + volume spike + above EMA20
            if (close[i] > high_20[i] and volume_spike[i] and 
                close[i] > ema_20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: breakdown below low_20 + volume spike + below EMA20
            elif (close[i] < low_20[i] and volume_spike[i] and 
                  close[i] < ema_20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: breakdown below low_20 OR close below EMA20
            if (close[i] < low_20[i]) or (close[i] < ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: breakout above high_20 OR close above EMA20
            if (close[i] > high_20[i]) or (close[i] > ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals