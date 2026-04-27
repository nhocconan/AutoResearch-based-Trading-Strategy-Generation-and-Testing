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
    
    # Get 1d data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.mean(tr[i-14:i])
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200
    ema_period = 200
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                        ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, EMA, and volume MA
    start_idx = max(14, 200, vol_period) + 10  # extra buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: Price > weekly EMA200 + volume spike + price > recent high
            if (price > ema_1w_aligned[i] and 
                vol_ratio > 1.5 and 
                price > high[i-1]):
                signals[i] = size
                position = 1
            # Short: Price < weekly EMA200 + volume spike + price < recent low
            elif (price < ema_1w_aligned[i] and 
                  vol_ratio > 1.5 and 
                  price < low[i-1]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price < weekly EMA200 OR ATR-based stop
            if (price < ema_1w_aligned[i] or 
                price < high[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price > weekly EMA200 OR ATR-based stop
            if (price > ema_1w_aligned[i] or 
                price > low[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WeeklyEMA200_Breakout_Volume_Spike"
timeframe = "4h"
leverage = 1.0