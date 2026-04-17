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
    
    # === 1d SMA (20-period) for trend direction ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate SMA with proper minimum period
    sma_20_1d = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if i >= 19:
            sma_20_1d[i] = np.mean(close_1d[i-19:i+1])
    
    # Align to 4h timeframe
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    
    # === 1d ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    atr_14_1d = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 4h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # === 4h Donchian (20-period) for breakout detection ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels
    upper_20_4h = np.full_like(high_4h, np.nan)
    lower_20_4h = np.full_like(low_4h, np.nan)
    for i in range(len(high_4h)):
        if i >= 19:
            upper_20_4h[i] = np.max(high_4h[i-19:i+1])
            lower_20_4h[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian to 4h timeframe
    upper_20_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_20_4h)
    lower_20_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_20_4h)
    
    # === 4h Volume (20-period average) for confirmation ===
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20_4h[i] = np.mean(volume_4h[i-19:i+1])
    
    # Volume confirmation: current > 1.5x average
    vol_confirm_4h = volume_4h > vol_ma_20_4h * 1.5
    vol_confirm_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_confirm_4h.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_20_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(upper_20_4h_aligned[i]) or np.isnan(lower_20_4h_aligned[i]) or
            np.isnan(vol_confirm_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND in high volume regime (4h)
        if position == 0:
            # Long: price breaks above upper Donchian + above SMA200 + volatility filter + volume confirmation
            if (close[i] > upper_20_4h_aligned[i] and 
                close[i] > sma_20_1d_aligned[i] and
                atr_14_1d_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm_4h_aligned[i] > 0.5):  # high volume regime
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian + below SMA200 + volatility filter + volume confirmation
            elif (close[i] < lower_20_4h_aligned[i] and 
                  close[i] < sma_20_1d_aligned[i] and
                  atr_14_1d_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm_4h_aligned[i] > 0.5):  # high volume regime
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower Donchian
            if close[i] < lower_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian
            if close[i] > upper_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_SMA200_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0