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
    
    # Calculate Donchian upper/lower
    upper_12h = np.full_like(high_12h, np.nan)
    lower_12h = np.full_like(low_12h, np.nan)
    for i in range(len(high_12h)):
        if i >= 19:
            upper_12h[i] = np.max(high_12h[i-19:i+1])
            lower_12h[i] = np.min(low_12h[i-19:i+1])
        elif i > 0:
            upper_12h[i] = np.max(high_12h[0:i+1])
            lower_12h[i] = np.min(low_12h[0:i+1])
        else:
            upper_12h[i] = high_12h[0]
            lower_12h[i] = low_12h[0]
    
    # === 12h EMA(50) for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = close_12h[i] * 0.0392 + ema_50_12h[i-1] * 0.9608  # alpha = 2/(50+1)
    
    # === 12h Volume spike detection ===
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = np.full_like(vol_12h, np.nan)
    for i in range(len(vol_12h)):
        if i >= 19:
            vol_ma_20_12h[i] = np.mean(vol_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20_12h[i] = np.mean(vol_12h[0:i+1])
        else:
            vol_ma_20_12h[i] = vol_12h[0]
    
    vol_spike_12h = vol_12h > (vol_ma_20_12h * 2.0)
    
    # === Align 12h indicators to 4h timeframe ===
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # === 4h ATR for volatility filter and stop management ===
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.concatenate([[close[0]], close[:-1]])), np.abs(low - np.concatenate([[close[0]], close[:-1]]))))
    atr = np.full_like(close, np.nan)
    for i in range(len(tr)):
        if i >= 13:
            atr[i] = np.mean(tr[i-13:i+1])
        elif i > 0:
            atr[i] = np.mean(tr[0:i+1])
        else:
            atr[i] = tr[0]
    
    # === 4h volume confirmation ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[0:i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    vol_confirm = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian in uptrend with volume spike
            if (close[i] > upper_12h_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_spike_12h_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian in downtrend with volume spike
            elif (close[i] < lower_12h_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_spike_12h_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price closes below EMA(50) or ATR-based stop
            if (close[i] < ema_50_12h_aligned[i]) or \
               (close[i] < (upper_12h_aligned[i] - 2.0 * atr[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above EMA(50) or ATR-based stop
            if (close[i] > ema_50_12h_aligned[i]) or \
               (close[i] > (lower_12h_aligned[i] + 2.0 * atr[i])):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA50_VolumeSpike_ATR_v1"
timeframe = "4h"
leverage = 1.0