#!/usr/bin/env python3
name = "4h_TrixVolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

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
    
    # 1h data for TRIX
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 30:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    
    # 1d data for regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TRIX (15-period triple EMA)
    ema1 = pd.Series(close_1h).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix_raw[0] = 0  # first value has no previous
    
    # Align TRIX to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1h, trix_raw)
    
    # Calculate Choppiness Index (14-period) on 1d
    atr_1d = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d[i] = tr
    atr_sum_1d = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum_1d / (highest_high_1d - lowest_low_1d)) / np.log10(14)
    chop = np.where((highest_high_1d - lowest_low_1d) == 0, 50, chop)
    
    # Align Choppiness to 4h
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Ensure TRIX and chop are ready
    
    for i in range(start_idx, n):
        if np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: TRIX crosses above zero, chop > 50 (range), volume spike
            if (trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and 
                chop_aligned[i] > 50 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero, chop > 50 (range), volume spike
            elif (trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and 
                  chop_aligned[i] > 50 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero or chop < 30 (trending)
            if trix_aligned[i] < 0 or chop_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero or chop < 30 (trending)
            if trix_aligned[i] > 0 or chop_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals