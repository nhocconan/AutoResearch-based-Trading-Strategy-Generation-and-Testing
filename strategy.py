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
    
    # Load 12h data for regime and volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR(14) for volatility filter
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr_12h[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    atr_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 14:
        atr_12h[13] = np.mean(tr_12h[:14])
        for i in range(14, len(df_12h)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h volume average (20-period)
    vol_ma_20_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= 20:
        for i in range(19, len(df_12h)):
            vol_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Load 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian(20) on 4h
    donch_high_4h = np.full(len(df_4h), np.nan)
    donch_low_4h = np.full(len(df_4h), np.nan)
    
    for i in range(19, len(df_4h)):
        donch_high_4h[i] = np.max(high_4h[i-19:i+1])
        donch_low_4h[i] = np.min(low_4h[i-19:i+1])
    
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(donch_high_4h_aligned[i]) or
            np.isnan(donch_low_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        if atr_12h_aligned[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: require volume spike on 12h
        if vol_ma_20_12h_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20_12h_aligned[i]
        
        vol_threshold = 1.8
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation
            if (close[i] > donch_high_4h_aligned[i] and 
                volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume confirmation
            elif (close[i] < donch_low_4h_aligned[i] and 
                  volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: price breaks below Donchian low
            if close[i] < donch_low_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: price breaks above Donchian high
            if close[i] > donch_high_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Donchian_Volume_Breakout"
timeframe = "4h"
leverage = 1.0