#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Direction_Trend_Filter_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA on daily close - Kaufman Adaptive Moving Average
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0) if len(close_1d) > 1 else np.array([1.0])
    # Handle array shapes properly
    if len(change) < len(close_1d):
        change = np.pad(change, (10-1, 0), mode='edge')
    er = np.where(vol > 0, change / vol, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # 2-period EMA
    slow_sc = 2 / (30 + 1)  # 30-period EMA
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) > 0:
        kama[0] = close_1d[0]
        for i in range(1, len(close_1d)):
            if not np.isnan(sc[i-1]):
                kama[i] = kama[i-1] + sc[i-1] * (close_1d[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA AND above 12h EMA50 + volume confirmation
            if (close[i] > kama_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND below 12h EMA50 + volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below KAMA OR below 12h EMA50
            if close[i] < kama_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above KAMA OR above 12h EMA50
            if close[i] > kama_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals