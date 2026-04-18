#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_Filter
Hypothesis: Use 12h KAMA to filter trend direction, with RSI(14) < 30 for long and > 70 for short entries. KAMA adapts to market noise, reducing false signals in choppy markets. RSI extremes provide mean-reversion entries within the trend. Designed for low trade frequency on 12h timeframe to minimize fee drift while capturing high-probability reversals in both bull and bear markets.
"""

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
    
    # 12h KAMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Efficiency Ratio for KAMA
    change = np.abs(np.diff(close_12h, k=10))  # 10-period change
    abs_change = np.sum(np.abs(np.diff(close_12h)), axis=0)  # placeholder, will fix below
    
    # Proper ER calculation
    er = np.zeros_like(close_12h)
    for i in range(10, len(close_12h)):
        direction = np.abs(close_12h[i] - close_12h[i-10])
        volatility = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
        if volatility > 0:
            er[i] = direction / volatility
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # seed
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 10)  # RSI and KAMA seed
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: price above KAMA and RSI oversold
            if price > kama_val and rsi_val < 30:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and RSI overbought
            elif price < kama_val and rsi_val > 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below KAMA or RSI overbought
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above KAMA or RSI oversold
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0