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
    
    # Get 1d data for ATR and close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ATR with proper handling
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[np.nan], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr[1:]])  # Align with original indexing
    
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.mean(tr[i-14:i])
    
    # Get 1d data for Donchian channels (20-period)
    donch_len = 20
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    for i in range(donch_len, len(high_1d)):
        upper[i] = np.max(high_1d[i-donch_len:i])
        lower[i] = np.min(low_1d[i-donch_len:i])
    
    # Get 12h data for trend filter (EMA 50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align all indicators to 12h timeframe
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    upper_12h = align_htf_to_ltf(prices, df_1d, upper)
    lower_12h = align_htf_to_ltf(prices, df_1d, lower)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 30
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators ready
    start_idx = max(14, donch_len, vol_period, ema_period) + 10
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_12h[i]) or np.isnan(upper_12h[i]) or 
            np.isnan(lower_12h[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_12h[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume spike + above 12h EMA50
            if (price > upper_12h[i] and 
                vol_ratio > 1.5 and 
                price > ema_12h_aligned[i]):
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian + volume spike + below 12h EMA50
            elif (price < lower_12h[i] and 
                  vol_ratio > 1.5 and 
                  price < ema_12h_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below 12h EMA50 OR ATR-based stop
            if (price < ema_12h_aligned[i] or 
                price < high[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above 12h EMA50 OR ATR-based stop
            if (price > ema_12h_aligned[i] or 
                price > low[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_12hEMA50_Volume"
timeframe = "12h"
leverage = 1.0