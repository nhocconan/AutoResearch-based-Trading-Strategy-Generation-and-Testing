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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day closing prices
    close_1d = df_1d['close'].values
    
    # Calculate 1-day Donchian channel (20-period)
    upper = np.full(len(close_1d), np.nan)
    lower = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            upper[i] = np.max(close_1d[i-19:i+1])
            lower[i] = np.min(close_1d[i-19:i+1])
    
    # Calculate 1-day ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_prev)
    tr3 = np.abs(low_1d - close_1d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1-day EMA (50-period) for trend filter
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        alpha = 2 / (50 + 1)
        ema_50_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    
    # Align 1d indicators to 4h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above EMA50 trend
            if price > upper_1d_aligned[i] and vol_filter and price > ema_50_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below lower Donchian with volume and below EMA50 trend
            elif price < lower_1d_aligned[i] and vol_filter and price < ema_50_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below lower Donchian or volatility spike (potential reversal)
            if price < lower_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above upper Donchian or volatility spike (potential reversal)
            if price > upper_1d_aligned[i] or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0