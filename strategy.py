#!/usr/bin/env python3
"""
12h_KAMA_Trend_Reversal_1dFilt
Hypothesis: 12h KAMA trend with 1d trend filter and volume confirmation. 
KAMA adapts to market efficiency, reducing whipsaw in ranging markets while capturing trends. 
1d trend filter ensures alignment with higher timeframe momentum. Volume surge confirms institutional participation.
Works in both bull and bear markets by adapting trend sensitivity and requiring volume confirmation.
Target: 15-35 trades/year.
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
    volume = prices['volume'].values
    
    # Calculate KAMA on price
    # Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, axis=0)), axis=0) if hasattr(np.diff(close, axis=0), 'shape') else np.sum(np.abs(np.diff(close)), axis=0)
    # Simplified ER calculation for vectorization
    er = np.zeros_like(close)
    for i in range(10, n):
        price_change = np.abs(close[i] - close[i-10])
        price_volatility = np.sum(np.abs(np.diff(close[i-9:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after first 10 periods
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align higher timeframe data to 12h
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: price > KAMA and price > daily EMA50 and volume surge
        long_entry = (close[i] > kama_aligned[i] and 
                     close[i] > ema_50_1d_aligned[i] and 
                     volume_surge[i])
        
        # Short: price < KAMA and price < daily EMA50 and volume surge
        short_entry = (close[i] < kama_aligned[i] and 
                      close[i] < ema_50_1d_aligned[i] and 
                      volume_surge[i])
        
        # Exit when price crosses KAMA in opposite direction
        long_exit = close[i] < kama_aligned[i]
        short_exit = close[i] > kama_aligned[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_Reversal_1dFilt"
timeframe = "12h"
leverage = 1.0