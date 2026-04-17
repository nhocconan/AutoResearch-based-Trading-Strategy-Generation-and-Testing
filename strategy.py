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
    
    # === 12h Donchian Channel (20-period) for trend direction ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper and lower bands
    upper_12h = np.full_like(high_12h, np.nan)
    lower_12h = np.full_like(low_12h, np.nan)
    for i in range(len(high_12h)):
        if i >= 19:
            upper_12h[i] = np.max(high_12h[i-19:i+1])
            lower_12h[i] = np.min(low_12h[i-19:i+1])
        elif i > 0:
            upper_12h[i] = np.max(high_12h[max(0, i-9):i+1])
            lower_12h[i] = np.min(low_12h[max(0, i-9):i+1])
        else:
            upper_12h[i] = high_12h[0]
            lower_12h[i] = low_12h[0]
    
    # Align Donchian bands to 4h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    
    # === 12h ATR (14-period) for volatility filter ===
    high_12h_arr = df_12h['high'].values
    low_12h_arr = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    # True Range
    tr1 = high_12h_arr - low_12h_arr
    tr2 = np.abs(high_12h_arr - np.roll(close_12h_arr, 1))
    tr3 = np.abs(low_12h_arr - np.roll(close_12h_arr, 1))
    tr1[0] = high_12h_arr[0] - low_12h_arr[0]
    tr2[0] = np.abs(high_12h_arr[0] - close_12h_arr[0])
    tr3[0] = np.abs(low_12h_arr[0] - close_12h_arr[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
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
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume confirmation
        if position == 0:
            # Long: Price breaks above 12h Donchian upper + volatility filter + volume confirmation
            if (close[i] > upper_12h_aligned[i] and 
                atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below 12h Donchian lower + volatility filter + volume confirmation
            elif (close[i] < lower_12h_aligned[i] and 
                  atr_14_aligned[i] > 0.005 * close[i] and  # volatility filter
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Price closes below 12h Donchian lower
            if close[i] < lower_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above 12h Donchian upper
            if close[i] > upper_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian12h_Breakout_Volume_VolatilityFilter_v1"
timeframe = "4h"
leverage = 1.0