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
    
    # Calculate upper and lower bands
    upper_20 = np.full_like(high_1d, np.nan)
    lower_20 = np.full_like(low_1d, np.nan)
    
    for i in range(len(high_1d)):
        if i >= 19:
            upper_20[i] = np.max(high_1d[i-19:i+1])
            lower_20[i] = np.min(low_1d[i-19:i+1])
        elif i > 0:
            upper_20[i] = np.max(high_1d[max(0, i-9):i+1])
            lower_20[i] = np.min(low_1d[max(0, i-9):i+1])
        else:
            upper_20[i] = high_1d[0]
            lower_20[i] = low_1d[0]
    
    # === 1d ATR (14-period) for volatility filter ===
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close[1:]) if len(close) > 1 else np.array([])
    tr3 = np.abs(low_1d[1:] - close[1:]) if len(close) > 1 else np.array([])
    tr = np.maximum(tr1, np.maximum(tr2, tr3)) if len(tr1) > 0 else np.array([])
    tr = np.concatenate([[np.nan], tr]) if len(tr) > 0 else np.array([np.nan])
    
    atr_14 = np.full_like(high_1d, np.nan)
    for i in range(len(high_1d)):
        if i >= 13:
            atr_14[i] = np.mean(tr[i-13:i+1])
        elif i > 0:
            atr_14[i] = np.mean(tr[1:i+1]) if len(tr[1:i+1]) > 0 else np.nan
        else:
            atr_14[i] = np.nan
    
    # === 1d Volume moving average (20-period) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        elif i > 0:
            vol_ma_20[i] = np.mean(volume[max(0, i-9):i+1])
        else:
            vol_ma_20[i] = volume[0]
    
    # === Align indicators to 12h timeframe ===
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above upper band with volume confirmation
            if (close[i] > upper_20_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below lower band with volume confirmation
            elif (close[i] < lower_20_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to lower band
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to upper band
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeBreakout_v1"
timeframe = "12h"
leverage = 1.0