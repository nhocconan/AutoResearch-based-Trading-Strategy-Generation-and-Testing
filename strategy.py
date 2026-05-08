#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Breakout_RSI_Confirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - trend direction
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period volatility
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period) on daily
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Align KAMA and RSI to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # enough for indicators to warm up
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA + RSI > 50 + volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA + RSI < 50 + volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below KAMA OR RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above KAMA OR RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals