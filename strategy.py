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
    
    # === 1w Donchian Channel (20-period) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian upper and lower bands
    upper = np.full_like(high_1w, np.nan)
    lower = np.full_like(low_1w, np.nan)
    period = 20
    for i in range(len(high_1w)):
        if i >= period - 1:
            upper[i] = np.max(high_1w[i-(period-1):i+1])
            lower[i] = np.min(low_1w[i-(period-1):i+1])
        elif i > 0:
            upper[i] = np.max(high_1w[0:i+1])
            lower[i] = np.min(low_1w[0:i+1])
        else:
            upper[i] = high_1w[0]
            lower[i] = low_1w[0]
    
    # === 1d EMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # seed
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = alpha * close_1d[i] + (1 - alpha) * ema_50[i-1]
    else:
        for i in range(len(close_1d)):
            ema_50[i] = np.mean(close_1d[:i+1]) if i >= 0 else close_1d[0]
    
    # === 1d Volume confirmation (20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_20 = np.full_like(vol_1d, np.nan)
    for i in range(len(vol_1d)):
        if i >= 19:
            vol_ma_20[i] = np.mean(vol_1d[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(vol_1d[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = vol_1d[0]
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_spike = vol_1d > vol_ma_20 * 2.0
    
    # === Align indicators to daily timeframe ===
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly Donchian upper AND above daily EMA50 AND volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly Donchian lower AND below daily EMA50 AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price closes below weekly Donchian lower
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above weekly Donchian upper
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_EMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0