#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_KAMA_Trend_Volume_Confirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Get weekly data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA calculation on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period volatility
    # Fix dimensions
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    er = np.nan_to_num(er, nan=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Weekly ATR for regime filter (volatility regime)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_1w = np.full_like(tr, np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_1w[i] = np.mean(tr[1:15])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA + low volatility regime + volume confirmation
            if (close[i] > kama_1d_aligned[i] and 
                atr_1w_aligned[i] < np.nanpercentile(atr_1w_aligned[max(0, i-100):i+1], 50) and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + low volatility regime + volume confirmation
            elif (close[i] < kama_1d_aligned[i] and 
                  atr_1w_aligned[i] < np.nanpercentile(atr_1w_aligned[max(0, i-100):i+1], 50) and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA
            if close[i] < kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA
            if close[i] > kama_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals