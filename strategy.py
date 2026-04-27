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
    
    # Get 1d data for ATR
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
    
    atr_14 = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_14[i] = np.nanmean(tr[i-13:i+1])
    
    # Get 1w data for weekly trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20
    ema_period = 20
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get 1d data for daily EMA20
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA20
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to primary timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 2.0x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, EMAs, and volume MA
    start_idx = max(14, ema_period, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_14_aligned[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price > daily EMA20 AND > weekly EMA20 AND volume spike
            if (price > ema_1d_aligned[i] and 
                price > ema_1w_aligned[i] and 
                vol_ratio > 2.0):
                signals[i] = size
                position = 1
            # Short: Price < daily EMA20 AND < weekly EMA20 AND volume spike
            elif (price < ema_1d_aligned[i] and 
                  price < ema_1w_aligned[i] and 
                  vol_ratio > 2.0):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price < daily EMA20 OR ATR-based stop
            if (price < ema_1d_aligned[i] or 
                price < ema_1d_aligned[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price > daily EMA20 OR ATR-based stop
            if (price > ema_1d_aligned[i] or 
                price > ema_1d_aligned[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1D_DualEMA_Volume_ATR_Trend"
timeframe = "1d"
leverage = 1.0