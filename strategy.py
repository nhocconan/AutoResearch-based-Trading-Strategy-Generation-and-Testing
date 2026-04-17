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
    
    # === 1d Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian upper and lower bands
    donchian_upper = np.full_like(high_1d, np.nan)
    donchian_lower = np.full_like(low_1d, np.nan)
    period = 20
    for i in range(len(high_1d)):
        if i >= period - 1:
            donchian_upper[i] = np.max(high_1d[i-(period-1):i+1])
            donchian_lower[i] = np.min(low_1d[i-(period-1):i+1])
        elif i > 0:
            donchian_upper[i] = np.max(high_1d[0:i+1])
            donchian_lower[i] = np.min(low_1d[0:i+1])
        else:
            donchian_upper[i] = high_1d[0]
            donchian_lower[i] = low_1d[0]
    
    # === 1d EMA(34) for trend filter ===
    ema_34 = np.full_like(df_1d['close'].values, np.nan)
    close_1d = df_1d['close'].values
    if len(close_1d) >= 34:
        ema_34[33] = np.mean(close_1d[:34])  # seed
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34[i] = alpha * close_1d[i] + (1 - alpha) * ema_34[i-1]
    else:
        for i in range(len(close_1d)):
            ema_34[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # === Align indicators to 4h timeframe ===
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === 4h Volume confirmation ===
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
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Close breaks above Donchian upper band AND volume confirmation AND price above EMA34
            if (close[i] > donchian_upper_aligned[i] and 
                vol_confirm[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Close breaks below Donchian lower band AND volume confirmation AND price below EMA34
            elif (close[i] < donchian_lower_aligned[i] and 
                  vol_confirm[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Close crosses below Donchian lower band OR volume drops below average
            if (close[i] < donchian_lower_aligned[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses above Donchian upper band OR volume drops below average
            if (close[i] > donchian_upper_aligned[i]) or (not vol_confirm[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA34_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0