#!/usr/bin/env python3
name = "12h_Donchian_20_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

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
    
    # 1d EMA50 trend filter (more robust than 34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels on 12h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need EMA and Donchian warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) 
            or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: break above upper Donchian in uptrend with volume
            if close[i] > upper[i] and close[i] > ema_50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian in downtrend with volume
            elif close[i] < lower[i] and close[i] < ema_50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through Donchian channel or trend fails
            if position == 1 and (close[i] < lower[i] or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (close[i] > upper[i] or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals