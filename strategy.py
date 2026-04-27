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
    
    # Get daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels
    upper_20 = np.full(len(df_1d), np.nan)
    lower_20 = np.full(len(df_1d), np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            upper_20[i] = np.max(high_1d[i-19:i+1])
            lower_20[i] = np.min(low_1d[i-19:i+1])
        else:
            upper_20[i] = np.max(high_1d[:i+1])
            lower_20[i] = np.min(low_1d[:i+1])
    
    # Align Donchian channels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(tr_1d)):
        if i < 13:
            atr_1d[i] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[i]
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h ATR(14) for breakout strength
    tr1_6h = high[1:] - low[1:]
    tr2_6h = np.abs(high[1:] - close[:-1])
    tr3_6h = np.abs(low[1:] - close[:-1])
    tr_6h = np.concatenate([[high[0] - low[0]], np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))])
    
    atr_6h = np.zeros(n)
    for i in range(n):
        if i < 13:
            atr_6h[i] = np.mean(tr_6h[:i+1]) if i > 0 else tr_6h[i]
        else:
            atr_6h[i] = (atr_6h[i-1] * 13 + tr_6h[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 20)  # Donchian needs 20, ATR needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20_aligned[i]) or
            np.isnan(lower_20_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 1.8x average volume
        volume_confirmation = vol_ratio > 1.8
        
        # ATR volatility filter: only trade when 6h ATR is above 40% of daily ATR
        vol_filter = atr_6h[i] > atr_1d_aligned[i] * 0.4
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and volatility
            if volume_confirmation and vol_filter and price > upper_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and volatility
            elif volume_confirmation and vol_filter and price < lower_20_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below midpoint or volatility drops
            midpoint = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if price < midpoint or atr_6h[i] < atr_1d_aligned[i] * 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price crosses above midpoint or volatility drops
            midpoint = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if price > midpoint or atr_6h[i] < atr_1d_aligned[i] * 0.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "6h_Donchian20_Breakout_VolumeVolFilter_v1"
timeframe = "6h"
leverage = 1.0