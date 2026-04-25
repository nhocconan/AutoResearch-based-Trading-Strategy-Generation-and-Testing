#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ATR Filter (Long-Only)
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirms institutional participation.
ATR filter ensures sufficient volatility. Long-only bias works in bull via breakouts and in bear via quick exits
to avoid large drawdowns. Target: 20-40 trades/year on 4h.
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
    
    # Donchian channel (20-period high/low)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR filter: ensure sufficient volatility
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14 > (atr_ma * 0.8)  # Trade when volatility is above 80% of its 50-period MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    # Start index: need enough for all indicators
    start_idx = max(lookback, 20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        atr_ok = atr_filter[i]
        
        # Donchian breakout conditions (long only)
        breakout_long = curr_close > highest_high[i-1]  # Break above previous period's high
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume + ATR filter
            long_entry = breakout_long and vol_spike and atr_ok
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price retouches Donchian low level
            if curr_close < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRFilter_LongOnly"
timeframe = "4h"
leverage = 1.0