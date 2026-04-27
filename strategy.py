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
    
    # Get daily data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily high and low for Donchian channel (20-day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day high and low (Donchian channel)
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Align daily Donchian channels to daily timeframe (1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d_arr = high_1d
    low_1d_arr = low_1d
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d_arr[1:] - low_1d_arr[1:], 
                     np.abs(high_1d_arr[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d_arr[1:] - close_1d[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    
    atr = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate 20-period volume average on daily
    vol_1d = df_1d['volume'].values
    vol_ma = np.full(len(vol_1d), np.nan)
    vol_period = 20
    for i in range(vol_period, len(vol_1d)):
        vol_ma[i] = np.mean(vol_1d[i-vol_period:i])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period: need 20 for Donchian, 14 for ATR, 20 for volume MA
    start_idx = max(20, 14, 20) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_aligned[i] if vol_ma_aligned[i] > 0 else 0
        
        # Volume filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume
            if price > donchian_high_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below 20-day low with volume
            elif price < donchian_low_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below 20-day low or ATR stop
            if price < donchian_low_aligned[i] or price < donchian_high_aligned[i] - 2.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above 20-day high or ATR stop
            if price > donchian_high_aligned[i] or price > donchian_low_aligned[i] + 2.0 * atr_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_VolumeFilter_ATRStop"
timeframe = "1d"
leverage = 1.0