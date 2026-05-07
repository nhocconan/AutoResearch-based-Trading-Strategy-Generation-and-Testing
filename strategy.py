#!/usr/bin/env python3
name = "1d_KAMA_Trend_With_RSI_Momentum"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA on daily close
    def kama(close, er_len=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[er_len:] = change[er_len:] / np.maximum(volatility[er_len:], 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[er_len] = close[er_len]
        for i in range(er_len+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10, 2, 30)
    kama_dir = np.where(kama_val > np.roll(kama_val, 1), 1, -1)
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly trend: EMA20 on weekly close
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, price above weekly EMA20
            if kama_dir[i] == 1 and rsi[i] > 50 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, price below weekly EMA20
            elif kama_dir[i] == -1 and rsi[i] < 50 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA down or RSI < 40
            if kama_dir[i] == -1 or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up or RSI > 60
            if kama_dir[i] == 1 or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals