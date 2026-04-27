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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day ATR (14-period) for volatility
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
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
    
    # Calculate 1-day EMA (200-period) for long-term trend
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        alpha = 2 / (200 + 1)
        ema_200_1d[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            ema_200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_200_1d[i-1]
    
    # Calculate 1-day Williams %R (14-period)
    highest_high_14 = np.full(len(high_1d), np.nan)
    lowest_low_14 = np.full(len(low_1d), np.nan)
    for i in range(13, len(high_1d)):
        highest_high_14[i] = np.max(high_1d[i-13:i+1])
        lowest_low_14[i] = np.min(low_1d[i-13:i+1])
    
    williams_r = np.full(len(close_1d), np.nan)
    for i in range(13, len(close_1d)):
        if highest_high_14[i] != lowest_low_14[i]:
            williams_r[i] = (highest_high_14[i] - close_1d[i]) / (highest_high_14[i] - lowest_low_14[i]) * -100
    
    # Align 1d indicators to 6h timeframe
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 2-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 2
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(14, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + above EMA200 trend + volume spike
            if williams_r_aligned[i] < -80 and price > ema_200_1d_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) + below EMA200 trend + volume spike
            elif williams_r_aligned[i] > -20 and price < ema_200_1d_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or volatility spike (potential reversal)
            if williams_r_aligned[i] > -20 or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or volatility spike (potential reversal)
            if williams_r_aligned[i] < -80 or (vol_ratio > 2.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsR_200EMA_Volume"
timeframe = "6h"
leverage = 1.0