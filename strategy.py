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
    
    # === 12h Donchian Channel (20-period) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper and lower bands
    upper = np.full_like(high_12h, np.nan)
    lower = np.full_like(low_12h, np.nan)
    period = 20
    for i in range(len(high_12h)):
        if i >= period - 1:
            upper[i] = np.max(high_12h[i-(period-1):i+1])
            lower[i] = np.min(low_12h[i-(period-1):i+1])
        elif i > 0:
            upper[i] = np.max(high_12h[0:i+1])
            lower[i] = np.min(low_12h[0:i+1])
        else:
            upper[i] = high_12h[0]
            lower[i] = low_12h[0]
    
    # === 12h EMA(34) for trend filter ===
    ema_34 = np.full_like(high_12h, np.nan)
    if len(high_12h) >= 34:
        ema_34[33] = np.mean(high_12h[:34])  # seed
        alpha = 2 / (34 + 1)
        for i in range(34, len(high_12h)):
            ema_34[i] = alpha * high_12h[i] + (1 - alpha) * ema_34[i-1]
    else:
        for i in range(len(high_12h)):
            ema_34[i] = np.mean(high_12h[:i+1]) if i >= 0 else high_12h[0]
    
    # === Align indicators to 12h timeframe ===
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    
    # === 12h Volume confirmation ===
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
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper band AND price above EMA34 AND volume confirmation
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower band AND price below EMA34 AND volume confirmation
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower band
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper band
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_EMA34_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0