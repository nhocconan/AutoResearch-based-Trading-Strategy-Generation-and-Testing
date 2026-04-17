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
    
    # === 12h Donchian Channel (20-period) for trend direction ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate upper and lower Donchian bands
    upper_dc = np.full_like(high_12h, np.nan)
    lower_dc = np.full_like(low_12h, np.nan)
    for i in range(len(high_12h)):
        if i >= 19:
            upper_dc[i] = np.max(high_12h[i-19:i+1])
            lower_dc[i] = np.min(low_12h[i-19:i+1])
        elif i > 0:
            upper_dc[i] = np.max(high_12h[max(0, i-9):i+1])
            lower_dc[i] = np.min(low_12h[max(0, i-9):i+1])
        else:
            upper_dc[i] = high_12h[0]
            lower_dc[i] = low_12h[0]
    
    # === 12h 20-period EMA for trend filter ===
    close_12h = df_12h['close'].values
    ema_20 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 20:
        ema_20[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_20[i] = (close_12h[i] * 2 + ema_20[i-1] * 18) / 20
    
    # === 4h Volume confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    
    # Calculate 20-period average volume on 4h timeframe
    vol_ma_20 = np.full_like(volume_4h, np.nan)
    for i in range(len(volume_4h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_4h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_4h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_4h[0]
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_confirm = volume_4h > vol_ma_20 * 1.5
    
    # === Align indicators to 4h timeframe ===
    upper_dc_aligned = align_htf_to_ltf(prices, df_12h, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_12h, lower_dc)
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: price breaks above 12h upper Donchian + above 12h EMA20 + volume confirmation
            if (close[i] > upper_dc_aligned[i] and 
                close[i] > ema_20_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 12h lower Donchian + below 12h EMA20 + volume confirmation
            elif (close[i] < lower_dc_aligned[i] and 
                  close[i] < ema_20_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below 12h EMA20
            if close[i] < ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 12h EMA20
            if close[i] > ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Donchian20_EMA20_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0