#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v3"
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
    
    # 1D trend filter (using daily close)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma * 1.2  # Require 20% above average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(sma_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian OR trend turns down
            if close[i] < low_20[i] or close[i] < sma_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian OR trend turns up
            if close[i] > high_20[i] or close[i] > sma_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation and clear trend
            if not volume_ok[i]:
                signals[i] = 0.0
                continue
                
            # Long: price breaks above upper Donchian + price above 1D SMA50
            if close[i] > high_20[i] and close[i] > sma_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian + price below 1D SMA50
            elif close[i] < low_20[i] and close[i] < sma_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals