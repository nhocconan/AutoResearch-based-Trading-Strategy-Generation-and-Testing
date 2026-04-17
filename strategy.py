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
    
    # === 4h Donchian Channel (20-period) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian upper/lower bands
    upper_4h = np.full_like(high_4h, np.nan)
    lower_4h = np.full_like(low_4h, np.nan)
    period = 20
    for i in range(len(high_4h)):
        if i >= period - 1:
            upper_4h[i] = np.max(high_4h[i-(period-1):i+1])
            lower_4h[i] = np.min(low_4h[i-(period-1):i+1])
        elif i > 0:
            upper_4h[i] = np.max(high_4h[0:i+1])
            lower_4h[i] = np.min(low_4h[0:i+1])
        else:
            upper_4h[i] = high_4h[0]
            lower_4h[i] = low_4h[0]
    
    # === Align Donchian bands to 4h timeframe ===
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # === 12h EMA(34) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34)
    ema_34_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])  # seed
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_34_12h[i-1]
    else:
        for i in range(len(close_12h)):
            ema_34_12h[i] = np.mean(close_12h[:i+1]) if i >= 0 else close_12h[0]
    
    # === Align EMA to 4h timeframe ===
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === Volume confirmation ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_confirm = volume > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above Donchian upper band AND above EMA34 AND volume confirmation
            if (close[i] > upper_4h_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below Donchian lower band AND below EMA34 AND volume confirmation
            elif (close[i] < lower_4h_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price crosses below Donchian lower band
            if close[i] < lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper band
            if close[i] > upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA34_VolumeFilter_Trend"
timeframe = "4h"
leverage = 1.0