#!/usr/bin/env python3
"""
12h_Donchian_Breakout_VolumeTrend_v1
Donchian(20) breakout + volume confirmation + EMA34 trend filter on 12h timeframe.
Long when price breaks above upper band with volume spike and EMA34 up.
Short when price breaks below lower band with volume spike and EMA34 down.
Designed to capture trend continuation with volatility expansion.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # === 12h Donchian Channel (20-period) ===
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(n):
        if i >= 19:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
        elif i > 0:
            highest_high[i] = np.max(high[0:i+1])
            lowest_low[i] = np.min(low[0:i+1])
        else:
            highest_high[i] = high[i]
            lowest_low[i] = low[i]
    
    # === 12h EMA34 for trend filter ===
    ema34 = np.full_like(close, np.nan)
    if n >= 34:
        ema34[33] = np.mean(close[0:34])
        for i in range(34, n):
            ema34[i] = (close[i] * 2 / (34 + 1)) + (ema34[i-1] * (32 / (34 + 1)))
    
    # === 12h Volume confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[0:i+1])
        else:
            vol_ma_20[i] = volume[i]
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema34[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper band, EMA34 rising, volume confirmation
            if (close[i] > highest_high[i] and 
                ema34[i] > ema34[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower band, EMA34 falling, volume confirmation
            elif (close[i] < lowest_low[i] and 
                  ema34[i] < ema34[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below EMA34 or opposite Donchian break
            if (close[i] < ema34[i] or 
                close[i] < lowest_low[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above EMA34 or opposite Donchian break
            if (close[i] > ema34[i] or 
                close[i] > highest_high[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0