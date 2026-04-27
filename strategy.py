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
    
    # Get 1d data for 100-bar EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 100 EMA on daily
    ema_period = 100
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        # Initialize with SMA
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                        ema_1d[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Get 4h data for ATR volatility filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) on 4h
    tr = np.zeros(len(close_4h))
    tr[0] = high_4h[0] - low_4h[0]
    for i in range(1, len(close_4h)):
        tr[i] = max(high_4h[i] - low_4h[i], 
                   abs(high_4h[i] - close_4h[i-1]), 
                   abs(low_4h[i] - close_4h[i-1]))
    
    atr_period = 14
    atr_4h = np.full(len(close_4h), np.nan)
    if len(tr) >= atr_period:
        atr_4h[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr_4h[i] = (tr[i] * (1 / atr_period) + 
                        atr_4h[i-1] * (1 - 1 / atr_period))
    
    # Align indicators to 4h timeframe
    ema_1d_aligned_4h = align_htf_to_ltf(df_4h['open_time'].values, df_1d, ema_1d)
    atr_4h_aligned_4h = atr_4h  # Already on 4h timeframe
    
    # Align from 4h to 15m
    ema_1d_aligned = align_htf_to_ltf(prices, df_4h, ema_1d_aligned_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_aligned_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA, ATR, and volume MA
    start_idx = max(100, 14, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price above daily EMA100 + volume spike + volatility filter
            if (price > ema_1d_aligned[i] and 
                vol_ratio > 1.5 and 
                atr_4h_aligned[i] > 0):
                signals[i] = size
                position = 1
            # Short: Price below daily EMA100 + volume spike + volatility filter
            elif (price < ema_1d_aligned[i] and 
                  vol_ratio > 1.5 and 
                  atr_4h_aligned[i] > 0):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below daily EMA100 OR volatility drops
            if (price < ema_1d_aligned[i] or 
                atr_4h_aligned[i] < atr_4h_aligned[i-1] * 0.8):  # Volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above daily EMA100 OR volatility drops
            if (price > ema_1d_aligned[i] or 
                atr_4h_aligned[i] < atr_4h_aligned[i-1] * 0.8):  # Volatility contraction
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "15m_DailyEMA100_Volume_VolatilityFilter"
timeframe = "15m"
leverage = 1.0