#!/usr/bin/env python3
"""
12h_1w_1d_Donchian_Breakout_Volume_Signal
Hypothesis: Long when price breaks above 1d Donchian(20) high with 1w trend up (price > SMA50) and volume confirmation; short when price breaks below 1d Donchian(20) low with 1w trend down (price < SMA50) and volume confirmation. Exit on opposite Donchian break or trend reversal. Works in bull/bear by following 1w trend and using volatility-based breakouts.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high_1d, 20)
    donchian_low = rolling_min(low_1d, 20)
    
    # Align to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Load 1w data for trend (SMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(sma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above Donchian high with uptrend and volume
            if (price > donchian_high_aligned[i] and 
                price > sma_50_aligned[i] and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low with downtrend and volume
            elif (price < donchian_low_aligned[i] and 
                  price < sma_50_aligned[i] and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low or trend reversal
            if price < donchian_low_aligned[i] or price < sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high or trend reversal
            if price > donchian_high_aligned[i] or price > sma_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_1d_Donchian_Breakout_Volume_Signal"
timeframe = "12h"
leverage = 1.0