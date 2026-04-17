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
    
    # === 1d Close (for weekly pivot) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === Weekly High/Low/Close (for pivot) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Calculate Weekly Pivot (Classic: P = (H+L+C)/3) ===
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # === 6h EMA (21) ===
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 6h Volume Average (20) ===
    volume_s = pd.Series(volume)
    vol_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # === Align indicators to 6h timeframe ===
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)  # EMA on weekly close
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above weekly pivot AND above EMA21 AND volume > 20-period average
            if (close[i] > pivot_1w_aligned[i] and 
                close[i] > ema_21_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price below weekly pivot AND below EMA21 AND volume > 20-period average
            elif (close[i] < pivot_1w_aligned[i] and 
                  close[i] < ema_21_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below weekly pivot OR below EMA21
            if (close[i] < pivot_1w_aligned[i] or 
                close[i] < ema_21_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly pivot OR above EMA21
            if (close[i] > pivot_1w_aligned[i] or 
                close[i] > ema_21_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_EMA21_VolumeFilter"
timeframe = "6h"
leverage = 1.0