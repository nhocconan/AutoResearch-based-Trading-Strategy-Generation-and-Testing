#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper channel, 12-hour EMA is rising, and volume exceeds SMA.
Short when price breaks below Donchian lower channel, 12-hour EMA is falling, and volume exceeds SMA.
Exit when price reverses to the opposite Donchian channel or EMA trend changes direction.
This captures breakouts in trending markets while filtering by higher timeframe trend and volume.
Designed for low trade frequency by requiring multiple confirmations and breakout conditions.
Works in both bull and bear markets by following 12-hour trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 20-period SMA
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_sma[i]) or 
            np.isnan(ema20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian, 12h EMA rising, volume above average
            if (close[i] > high_20[i] and 
                ema20_12h_aligned[i] > ema20_12h_aligned[i-1] and 
                volume[i] > vol_sma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian, 12h EMA falling, volume above average
            elif (close[i] < low_20[i] and 
                  ema20_12h_aligned[i] < ema20_12h_aligned[i-1] and 
                  volume[i] > vol_sma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below lower Donchian OR 12h EMA turns down
                if (close[i] < low_20[i] or 
                    ema20_12h_aligned[i] < ema20_12h_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above upper Donchian OR 12h EMA turns up
                if (close[i] > high_20[i] or 
                    ema20_12h_aligned[i] > ema20_12h_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_12hEMA20_Volume"
timeframe = "4h"
leverage = 1.0