#!/usr/bin/env python3
"""
12h_1d_KAMA_RSI_Trend_v1
Hypothesis: On 12h timeframe, use KAMA (Kaufman Adaptive Moving Average) for trend direction and RSI for
overbought/oversold signals within the trend. Uses 1d trend filter (price > 1d EMA50 for longs,
price < 1d EMA50 for shorts) to avoid counter-trend trades. Designed for 15-35 trades/year by requiring
multiple confluence factors: KAMA trend alignment, RSI extreme, and 1d trend filter.
Works in bull markets via long entries in uptrends and in bear markets via short entries in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_KAMA_RSI_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # 2/(fast+1)
    slow_sc = 2 / (30 + 1) # 2/(slow+1)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.diff(close, n=er_length, prepend=repeat_first_n(close, er_length)))
    er = np.where(dir != 0, change / dir, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 1d data ONCE for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA trend: price above/below KAMA
        above_kama = close[i] > kama[i]
        below_kama = close[i] < kama[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Trend filter from 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_entry = above_kama and rsi_oversold and above_ema
        short_entry = below_kama and rsi_overbought and below_ema
        
        # Exit conditions: opposite signal or RSI returns to neutral
        long_exit = below_kama or rsi[i] >= 50
        short_exit = above_kama or rsi[i] <= 50
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

def repeat_first_n(arr, n):
    """Helper to repeat first n elements for diff calculation"""
    if n <= 0:
        return np.array([])
    return np.repeat(arr[0], n) if len(arr) > 0 else np.array([])