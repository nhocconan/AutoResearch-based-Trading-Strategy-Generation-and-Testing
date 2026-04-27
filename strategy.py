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
    
    # Get daily data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = np.full(len(df_1d), np.nan)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i+1] = np.mean(tr_1d[:i+1]) if i > 0 else tr_1d[0]
        else:
            atr_1d[i+1] = (atr_1d[i] * 13 + tr_1d[i]) / 14
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d[1:])  # Skip first NaN
    
    # Calculate 4h ATR(14) for position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            atr_4h[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr_4h[i] = (atr_4h[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need volume MA and ATR arrays
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_4h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: trade only when daily ATR is above its 20-day average
        if i >= 20:
            atr_20_avg = np.mean(atr_1d_aligned[max(0, i-20):i+1])
            vol_filter = atr_1d_aligned[i] > atr_20_avg * 0.8
        else:
            vol_filter = True
        
        # Volume confirmation: > 2x average volume
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long: volatility expansion + volume spike
            if vol_filter and volume_confirmation:
                signals[i] = 0.25
                position = 1
            # Short: volatility expansion + volume spike
            elif vol_filter and volume_confirmation:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: volatility contraction
            if atr_4h[i] < atr_4h[max(0, i-5):i+1].mean() * 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: volatility contraction
            if atr_4h[i] < atr_4h[max(0, i-5):i+1].mean() * 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VolatilityExpansion_VolumeSpike"
timeframe = "4h"
leverage = 1.0