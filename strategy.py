#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day volume spike filter and EMA trend filter.
Long when price breaks above 20-period Donchian high on 12h with volume > 1.8x 20-period average volume and 12h EMA > 50-period EMA.
Short when price breaks below 20-period Donchian low on 12h with volume > 1.8x 20-period average volume and 12h EMA < 50-period EMA.
Exit when price returns to 12h 50-period EMA.
Designed for low trade frequency (~10-20/year) to avoid fee drag, with trend filter to avoid whipsaws.
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
    
    # EMA trend filter on 12h - calculate directly
    ema_fast = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Donchian channels on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average volume on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume and EMA confirmation
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.8 * vol_ma[i] and 
                ema_fast[i] > ema_slow[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume and EMA confirmation
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.8 * vol_ma[i] and 
                  ema_fast[i] < ema_slow[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to 50-period EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to or below 50-period EMA
                if close[i] <= ema_slow[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to or above 50-period EMA
                if close[i] >= ema_slow[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_Volume_EMATrend"
timeframe = "12h"
leverage = 1.0