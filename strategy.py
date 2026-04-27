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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (classic)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get weekly data for trend filter: EMA(34) on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) with proper initialization
    ema_1w_34 = np.full(len(df_1w), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_1w_34[i] = close_1w[i]
        elif i < 34:
            ema_1w_34[i] = np.mean(close_1w[:i+1])
        else:
            if np.isnan(ema_1w_34[i-1]):
                ema_1w_34[i] = np.mean(close_1w[i-33:i+1])
            else:
                ema_1w_34[i] = close_1w[i] * alpha + ema_1w_34[i-1] * (1 - alpha)
    
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    # Calculate volume spike detector (20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    volume_spike = np.full(n, False)
    volume_spike[20:] = volume[20:] > 1.5 * vol_ma[20:]
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_1w_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price touches S1 support + volume spike + weekly uptrend
            if (price <= s1_aligned[i] * 1.005 and  # Allow 0.5% tolerance
                volume_spike[i] and 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches R1 resistance + volume spike + weekly downtrend
            elif (price >= r1_aligned[i] * 0.995 and  # Allow 0.5% tolerance
                  volume_spike[i] and 
                  ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price reaches pivot or weekly trend turns down
            if (price >= pivot_aligned[i] * 0.995 or 
                ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price reaches pivot or weekly trend turns up
            if (price <= pivot_aligned[i] * 1.005 or 
                ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_PivotTouch_VolumeSpike_WeeklyEMA34_v1"
timeframe = "12h"
leverage = 1.0