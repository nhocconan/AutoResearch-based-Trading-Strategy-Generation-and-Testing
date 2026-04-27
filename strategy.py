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
    
    # Get daily data for 12h strategy
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily high/low for pivot levels (simple pivot)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot point: (H + L + C) / 3
    pivot_point = (high_1d + low_1d + close_1d) / 3
    
    # Support 1 and Resistance 1 levels
    s1 = 2 * pivot_point - high_1d
    r1 = 2 * pivot_point - low_1d
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Calculate 12-period ATR for volatility filter
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])  # 14-period ATR
    
    # Calculate 20-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(20, 20)  # Need 20 for ATR and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr_value = atr[i]
        
        # Skip if ATR is too low (avoid choppy markets)
        if atr_value < 0.0001 * price:  # Avoid division by zero or near-zero ATR
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long entry: price crosses above R1 with volume spike and ATR filter
            if price > r1_aligned[i] and vol_ratio > 1.5:
                signals[i] = size
                position = 1
            # Short entry: price crosses below S1 with volume spike and ATR filter
            elif price < s1_aligned[i] and vol_ratio > 1.5:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below pivot or ATR-based stop
            if price < pivot_aligned[i] or price < close[i-1] - 1.5 * atr_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price crosses above pivot or ATR-based stop
            if price > pivot_aligned[i] or price > close[i-1] + 1.5 * atr_value:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_ATR"
timeframe = "12h"
leverage = 1.0