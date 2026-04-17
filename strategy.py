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
    
    # === 1d Donchian Channels (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate upper and lower bands
    upper_donchian = np.full_like(high_1d, np.nan)
    lower_donchian = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            upper_donchian[i] = np.max(high_1d[i-19:i+1])
            lower_donchian[i] = np.min(low_1d[i-19:i+1])
        else:
            upper_donchian[i] = np.nan
            lower_donchian[i] = np.nan
    
    # === 1d ATR (14-period) for stop loss ===
    tr = np.maximum(high_1d - low_1d,
                    np.maximum(np.abs(high_1d - np.roll(close, 1)),
                               np.abs(low_1d - np.roll(close, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 14:
            if i == 0:
                atr[i] = tr[i]
            else:
                atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # === Align to 12h timeframe ===
    upper_donchian_12h = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_12h = align_htf_to_ltf(prices, df_1d, lower_donchian)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr)
    
    # === 12h Volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period average volume on 12h timeframe
    vol_ma_20 = np.full_like(volume_12h, np.nan)
    for i in range(len(volume_12h)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume_12h[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_confirm = volume_12h > vol_ma_20 * 1.5
    
    # === Align volume confirmation to main timeframe ===
    vol_confirm_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_donchian_12h[i]) or 
            np.isnan(lower_donchian_12h[i]) or 
            np.isnan(atr_12h[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation
            if close[i] > upper_donchian_12h[i] and vol_confirm_aligned[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian + volume confirmation
            elif close[i] < lower_donchian_12h[i] and vol_confirm_aligned[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic with ATR-based stop loss
        elif position == 1:
            # Exit long: price closes below lower Donchian OR stop loss hit
            if close[i] < lower_donchian_12h[i] or close[i] < (high[i] - 2.0 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above upper Donchian OR stop loss hit
            if close[i] > upper_donchian_12h[i] or close[i] > (low[i] + 2.0 * atr_12h[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_Volume_ATR_Stop_v1"
timeframe = "12h"
leverage = 1.0