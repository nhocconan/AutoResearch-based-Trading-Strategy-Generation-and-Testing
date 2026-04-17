#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1w SMA(50) for long-term trend ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    sma_50_1w = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i >= 49:
            sma_50_1w[i] = np.mean(close_1w[i-49:i+1])
    
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # === 1d Donchian Channel (20) for breakout signals ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high_20 = np.full_like(high_1d, np.nan)
    donchian_low_20 = np.full_like(low_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 19:
            donchian_high_20[i] = np.max(high_1d[i-19:i+1])
            donchian_low_20[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # === 1d Volume confirmation (20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = np.full_like(vol_1d, np.nan)
    for i in range(len(vol_1d)):
        if i >= 19:
            vol_ma_20_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_confirm = vol_1d > vol_ma_20_1d_aligned * 2.0  # Volume > 2x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly SMA + volume confirmation
            if (close[i] > donchian_high_20_aligned[i] and 
                close[i] > sma_50_1w_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below Donchian low + below weekly SMA + volume confirmation
            elif (close[i] < donchian_low_20_aligned[i] and 
                  close[i] < sma_50_1w_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or volume confirmation fails
        elif position == 1:
            # Exit long: Price breaks below Donchian low OR volume confirmation fails
            if (close[i] < donchian_low_20_aligned[i] or not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Donchian high OR volume confirmation fails
            if (close[i] > donchian_high_20_aligned[i] or not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0