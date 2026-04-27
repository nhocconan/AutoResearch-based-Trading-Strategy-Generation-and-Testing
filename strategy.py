#!/usr/bin/env python3
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        upper_1d[i] = np.max(high_1d[i-20:i])
        lower_1d[i] = np.min(low_1d[i-20:i])
    
    # Get 1d data for ATR
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1d[i] = np.mean(tr[1:15])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Get 1d data for volume MA
    vol_ma_1d = np.full(len(volume), np.nan)
    vol_period = 20
    for i in range(vol_period, len(volume)):
        vol_ma_1d[i] = np.mean(volume[i-vol_period:i])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    ema_period = 50
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                        ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ratio = np.full(n, np.nan)
    for i in range(vol_period, n):
        if vol_ma_1d_aligned[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_1d_aligned[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, ATR, volume MA, and EMA
    start_idx = max(20, 14, vol_period, 50) + 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d upper Donchian + volume spike + price > 1w EMA50
            if (price > upper_1d_aligned[i] and 
                vol_ratio[i] > 1.5 and 
                price > ema_1w_aligned[i]):
                signals[i] = size
                position = 1
            # Short: price breaks below 1d lower Donchian + volume spike + price < 1w EMA50
            elif (price < lower_1d_aligned[i] and 
                  vol_ratio[i] > 1.5 and 
                  price < ema_1w_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below 1d lower Donchian OR ATR-based stop
            if (price < lower_1d_aligned[i] or 
                price < high[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above 1d upper Donchian OR ATR-based stop
            if (price > upper_1d_aligned[i] or 
                price > low[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_EMA50"
timeframe = "4h"
leverage = 1.0