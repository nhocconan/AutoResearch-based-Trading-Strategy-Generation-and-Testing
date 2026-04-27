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
    
    # Get daily data for pivot calculations
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily classic pivot points: PP, R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(n):
        if i < 14:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate volume average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    # Align daily pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 14)  # volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        
        # Volume confirmation: > 2.0x average volume
        volume_confirmation = vol_ratio > 2.0
        
        # ATR volatility filter: avoid low volatility periods
        # Only trade when ATR is above 50% of its 50-period average
        if i >= 50:
            atr_avg = np.mean(atr[i-50:i+1])
            vol_filter = atr[i] > atr_avg * 0.5
        else:
            vol_filter = True  # No filter during warmup
        
        if position == 0:
            # Long: break above daily R1 with volume and volatility
            if volume_confirmation and vol_filter and price > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 with volume and volatility
            elif volume_confirmation and vol_filter and price < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to daily pivot or volatility drops significantly
            if price < pivot_aligned[i] or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price returns to daily pivot or volatility drops significantly
            if price > pivot_aligned[i] or atr[i] < np.mean(atr[max(0, i-50):i+1]) * 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "12h_1D_Pivot_R1_S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0