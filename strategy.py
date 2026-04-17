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
    
    # === 1d Donchian Channel (20) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period high and low
    donchian_high = np.full_like(high_1d, np.nan)
    donchian_low = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            donchian_high[i] = np.max(high_1d[max(0, i-9):i+1])
            donchian_low[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            donchian_high[i] = high_1d[0]
            donchian_low[i] = low_1d[0]
    
    # === Align to 12h timeframe ===
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1d Volume Spike Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_10 = np.full_like(volume_1d, np.nan)
    
    for i in range(len(volume_1d)):
        if i >= 9:
            vol_ma_10[i] = np.mean(volume_1d[i-9:i+1])
        elif i > 0:
            vol_ma_10[i] = np.mean(volume_1d[max(0, i-4):i+1])
        else:
            vol_ma_10[i] = volume_1d[0]
    
    # Volume spike: current volume > 2x 10-period average
    vol_spike = volume_1d > vol_ma_10 * 2.0
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # === 12h Close Price (for breakout detection) ===
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat AND volume spike
        if position == 0:
            # Long: break above Donchian high + volume spike
            if close[i] > donchian_high_aligned[i] and vol_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: break below Donchian low + volume spike
            elif close[i] < donchian_low_aligned[i] and vol_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: opposite break or loss of momentum
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_Breakout_v1"
timeframe = "12h"
leverage = 1.0