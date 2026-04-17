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
    
    # Calculate 20-period high and low
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    period = 20
    for i in range(len(high_1d)):
        if i >= period - 1:
            highest_high[i] = np.max(high_1d[i-(period-1):i+1])
            lowest_low[i] = np.min(low_1d[i-(period-1):i+1])
        elif i > 0:
            highest_high[i] = np.max(high_1d[0:i+1])
            lowest_low[i] = np.min(low_1d[0:i+1])
        else:
            highest_high[i] = high_1d[0]
            lowest_low[i] = low_1d[0]
    
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
    
    # === 1d Volume confirmation (volume spike) ===
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(df_1d['volume'].values, np.nan)
    vol_1d = df_1d['volume'].values
    for i in range(len(vol_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(vol_1d[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = vol_1d[0]
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter)
    vol_confirm = vol_1d > vol_ma_20 * 2.0
    
    # === Align indicators to 12h timeframe ===
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Close breaks above 20-day high AND price above EMA34 AND volume spike
            if (close[i] > highest_high_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                vol_confirm_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Close breaks below 20-day low AND price below EMA34 AND volume spike
            elif (close[i] < lowest_low_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  vol_confirm_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: Close crosses below EMA34 OR closes below 20-day low
            if (close[i] < ema_34_aligned[i]) or (close[i] < lowest_low_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses above EMA34 OR closes above 20-day high
            if (close[i] > ema_34_aligned[i]) or (close[i] > highest_high_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA34_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0