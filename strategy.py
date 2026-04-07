#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume and 1d Trend Filter
Long when price breaks above 20-period Donchian high with above-average volume and 1d close > SMA50
Short when price breaks below 20-period Donchian low with above-average volume and 1d close < SMA50
Exit when price breaks in opposite direction
Designed to work in both bull and bear markets via trend filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_volume_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Trend Filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # === Donchian Channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or np.isnan(sma_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check 1d trend
        uptrend_1d = close[i] > sma_50_1d_aligned[i]
        downtrend_1d = close[i] < sma_50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry conditions
            if close[i] > highest_high[i] and uptrend_1d:
                # Breakout above Donchian high in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and downtrend_1d:
                # Breakdown below Donchian low in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals