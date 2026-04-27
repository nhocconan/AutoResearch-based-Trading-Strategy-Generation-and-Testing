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
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.mean(tr[i-14:i])
    
    # Get 1d data for 200 EMA trend filter
    ema_period = 200
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                        ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get 1d data for Donchian channel (20-period)
    donch_period = 20
    upper = np.full(len(high_1d), np.nan)
    lower = np.full(len(low_1d), np.nan)
    for i in range(donch_period - 1, len(high_1d)):
        upper[i] = np.max(high_1d[i-donch_period+1:i+1])
        lower[i] = np.min(low_1d[i-donch_period+1:i+1])
    
    # Align indicators to 12h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, EMA, Donchian, volume MA
    start_idx = max(14, ema_period, donch_period, vol_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + price > 1d EMA200
            if (price > upper_aligned[i] and 
                vol_ratio > 1.5 and 
                price > ema_1d_aligned[i]):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower + volume spike + price < 1d EMA200
            elif (price < lower_aligned[i] and 
                  vol_ratio > 1.5 and 
                  price < ema_1d_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls below Donchian lower OR ATR-based stop
            if (price < lower_aligned[i] or 
                price < high[i-1] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price rises above Donchian upper OR ATR-based stop
            if (price > upper_aligned[i] or 
                price > low[i-1] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA200_VolumeSpike"
timeframe = "12h"
leverage = 1.0