#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_200MA_Volume_Filter
Hypothesis: Donchian channel breakout with 200-period MA trend filter and volume spike.
Works in bull markets by buying upward breakouts, in bear markets by selling downward breakouts.
Uses 4h timeframe with 1d volume filter to limit trades to ~20-30/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h 200-period SMA for trend filter
    sma200_4h = np.full_like(close_4h, np.nan)
    for i in range(200, len(close_4h)):
        sma200_4h[i] = np.mean(close_4h[i-200:i])
    sma200_4h_aligned = align_htf_to_ltf(prices, df_4h, sma200_4h)
    
    # 4h Donchian channel (20-period)
    upper = np.full_like(high_4h, np.nan)
    lower = np.full_like(low_4h, np.nan)
    for i in range(20, len(high_4h)):
        upper[i] = np.max(high_4h[i-20:i])
        lower[i] = np.min(low_4h[i-20:i])
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    
    # Load 1d volume data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    # 20-day volume SMA
    vol_sma20_1d = np.full_like(vol_1d, np.nan)
    for i in range(20, len(vol_1d)):
        vol_sma20_1d[i] = np.mean(vol_1d[i-20:i])
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(sma200_4h_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        volume_ok = volume > 1.5 * vol_sma20_1d_aligned[i]
        
        if position == 0:
            # Long: price above 200MA + break above upper Donchian + volume
            if (price > sma200_4h_aligned[i] and 
                price > upper_aligned[i] and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: price below 200MA + break below lower Donchian + volume
            elif (price < sma200_4h_aligned[i] and 
                  price < lower_aligned[i] and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below 200MA or breaks below lower Donchian
            if price < sma200_4h_aligned[i] or price < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above 200MA or breaks above upper Donchian
            if price > sma200_4h_aligned[i] or price > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_20_200MA_Volume_Filter"
timeframe = "4h"
leverage = 1.0