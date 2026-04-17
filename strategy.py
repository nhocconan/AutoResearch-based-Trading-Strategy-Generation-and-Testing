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
    
    # === 1d Donchian Channel (20-period) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate upper and lower bands
    upper = np.full_like(high_1d, np.nan)
    lower = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            upper[i] = np.max(high_1d[i-19:i+1])
            lower[i] = np.min(low_1d[i-19:i+1])
        else:
            upper[i] = np.max(high_1d[max(0, i-9):i+1]) if i > 0 else high_1d[0]
            lower[i] = np.min(low_1d[max(0, i-9):i+1]) if i > 0 else low_1d[0]
    
    # === 1d ATR (14-period) for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # === Align indicators to 12h timeframe ===
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        vol_ma_20 = np.zeros_like(volume)
        for j in range(len(volume)):
            if j >= 19:
                vol_ma_20[j] = np.mean(volume[j-19:j+1])
            else:
                vol_ma_20[j] = np.mean(volume[max(0, j-9):j+1]) if j > 0 else volume[0]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper Donchian + volatility filter + volume
            if (close[i] > upper_aligned[i] and 
                atr_14_aligned[i] > 0.003 * close[i] and  # volatility filter
                vol_confirm):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower Donchian + volatility filter + volume
            elif (close[i] < lower_aligned[i] and 
                  atr_14_aligned[i] > 0.003 * close[i] and  # volatility filter
                  vol_confirm):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below lower Donchian
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above upper Donchian
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "Donchian20_Volume_ATR_12h_v1"
timeframe = "12h"
leverage = 1.0