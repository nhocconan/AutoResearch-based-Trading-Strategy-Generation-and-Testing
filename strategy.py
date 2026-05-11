#!/usr/bin/env python3
name = "1d_1w_KAMA_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_len = 10
    fast = 2
    slow = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Using loop for clarity and correctness
    er = np.zeros(n)
    for i in range(er_len, n):
        price_change = np.abs(close[i] - close[i-er_len])
        price_volatility = np.sum(np.abs(np.diff(close[i-er_len:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-week trend filter (EMA 34 on weekly close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA and above weekly EMA + volume
            if close[i] > kama[i] and close[i] > ema_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below weekly EMA + volume
            elif close[i] < kama[i] and close[i] < ema_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or below weekly EMA
            if close[i] < kama[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA or above weekly EMA
            if close[i] > kama[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals