#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donch_high = np.full(len(high_1d), np.nan)
    donch_low = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-20:i])
        donch_low[i] = np.min(low_1d[i-20:i])
    
    # Get 1w data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema_200 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_200[199] = np.mean(close_1w[:200])
        for i in range(200, len(close_1w)):
            ema_200[i] = (close_1w[i] * 2 / 201) + (ema_200[i-1] * 199 / 201)
    
    # Align indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Volume filter: current volume > 1.8x 30-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 30
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, EMA, and volume MA
    start_idx = max(20, 200, vol_period) + 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + price > weekly EMA200
            if (price > donch_high_aligned[i] and 
                vol_ratio > 1.8 and 
                price > ema_200_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Price breaks below Donchian low + volume spike + price < weekly EMA200
            elif (price < donch_low_aligned[i] and 
                  vol_ratio > 1.8 and 
                  price < ema_200_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price breaks below Donchian low OR volume drops
            if (price < donch_low_aligned[i] or 
                vol_ratio < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price breaks above Donchian high OR volume drops
            if (price > donch_high_aligned[i] or 
                vol_ratio < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_WeeklyEMA200_Volume"
timeframe = "4h"
leverage = 1.0