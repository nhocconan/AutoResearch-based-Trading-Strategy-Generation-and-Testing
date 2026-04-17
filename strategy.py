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
    
    # === 1d Donchian channels (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate upper and lower Donchian channels
    donch_high = np.full_like(high_1d, np.nan)
    donch_low = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            donch_high[i] = np.max(high_1d[i-19:i+1])
            donch_low[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            donch_high[i] = np.max(high_1d[max(0, i-9):i+1])
            donch_low[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            donch_high[i] = high_1d[0]
            donch_low[i] = low_1d[0]
    
    # === 1d ATR (14-period) for volatility filter ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - np.concatenate([[high_1d[0]], high_1d[:-1]]))
    tr3 = np.abs(low_1d[1:] - np.concatenate([[low_1d[0]], low_1d[:-1]]))
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            if i == 0:
                atr_14[i] = tr[i]
            else:
                atr_14[i] = (atr_14[i-1] * i + tr[i]) / (i + 1)
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === Align indicators to 12h timeframe ===
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # === 12h Volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h timeframe
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume_12h[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume_12h[0]
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_confirm = volume_12h > vol_ma_20 * 1.5
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Breakout logic with volume confirmation
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation
            if close[i] > donch_high_aligned[i] and vol_confirm[i]:
                signals[i] = 0.30
                position = 1
                continue
            # Short: price breaks below Donchian low + volume confirmation
            elif close[i] < donch_low_aligned[i] and vol_confirm[i]:
                signals[i] = -0.30
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite breakout
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Donchian20_Breakout_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0