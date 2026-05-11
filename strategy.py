#!/usr/bin/env python3
name = "4h_KAMA_Direction_200MA_Filter_1dTrend"
timeframe = "4h"
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
    
    # 1d data for trend filter and KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 200-day EMA for trend filter (long-term trend)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d
    # KAMA parameters: ER period=10, Fast=2, Slow=30
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d)).sum()  # Will be calculated properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        direction = np.abs(close_1d[i] - close_1d[i-10])
        volatility = np.abs(np.diff(close_1d[i-9:i+1])).sum()
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Volume spike filter (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)  # Moderate threshold
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Ensure EMA200 and KAMA are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(kama_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, above 200 EMA, volume spike
            if (close[i] > kama_aligned[i] and 
                close[i] > ema200_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, below 200 EMA, volume spike
            elif (close[i] < kama_aligned[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below KAMA or below 200 EMA
            if close[i] < kama_aligned[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above KAMA or above 200 EMA
            if close[i] > kama_aligned[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals