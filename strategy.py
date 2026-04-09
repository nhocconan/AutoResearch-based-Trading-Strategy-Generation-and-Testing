#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_keltner_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Keltner Channel
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA and ATR for Keltner Channel
    close_d = df_d['close'].values
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    
    # Daily EMA(20)
    ema_20 = np.full(len(df_d), np.nan)
    if len(close_d) >= 20:
        ema_20[19] = np.mean(close_d[:20])
        for i in range(20, len(close_d)):
            ema_20[i] = close_d[i] * 0.1 + ema_20[i-1] * 0.9
    
    # Daily ATR(10)
    atr_10 = np.full(len(df_d), np.nan)
    if len(df_d) >= 11:
        tr = np.zeros(len(df_d))
        for i in range(1, len(df_d)):
            tr[i] = max(high_d[i] - low_d[i], 
                       abs(high_d[i] - close_d[i-1]), 
                       abs(low_d[i] - close_d[i-1]))
        for i in range(10, len(df_d)):
            atr_10[i] = np.mean(tr[i-9:i+1])
    
    # Keltner Channel: EMA(20) ± 2 * ATR(10)
    upper_keltner = np.full(len(df_d), np.nan)
    lower_keltner = np.full(len(df_d), np.nan)
    for i in range(len(df_d)):
        if not np.isnan(ema_20[i]) and not np.isnan(atr_10[i]):
            upper_keltner[i] = ema_20[i] + 2 * atr_10[i]
            lower_keltner[i] = ema_20[i] - 2 * atr_10[i]
    
    # Align daily Keltner levels to 6h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_d, ema_20)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_d, lower_keltner)
    
    # Volume confirmation: 3-period average (3*6h = 18h ~ 3/4 day)
    vol_ma_3 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 3:
            vol_sum -= volume[i-3]
        if i >= 2:
            vol_ma_3[i] = vol_sum / 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(vol_ma_3[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to EMA(20)
            if close[i] >= ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to EMA(20)
            if close[i] <= ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches lower Keltner band with volume confirmation
            vol_ratio = volume[i] / vol_ma_3[i] if vol_ma_3[i] > 0 else 0
            if (close[i] <= lower_keltner_aligned[i] and 
                vol_ratio > 1.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches upper Keltner band with volume confirmation
            elif (close[i] >= upper_keltner_aligned[i] and 
                  vol_ratio > 1.8):
                position = -1
                signals[i] = -0.25
    
    return signals