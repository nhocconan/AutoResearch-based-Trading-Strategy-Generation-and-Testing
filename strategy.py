#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
1d strategy using KAMA direction, RSI extremes, and Chop index regime filter.
- Long: KAMA trending up + RSI > 50 + Chop > 61.8 (range) → mean reversion long
- Short: KAMA trending down + RSI < 50 + Chop > 61.8 (range) → mean reversion short
- Exit: Opposite KAMA direction or Chop < 38.2 (trend)
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years)
Works in range-bound markets (2022-2024, 2025+) with mean reversion
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA (adaptive moving average)
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Fix: calculate volatility properly
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d = kama
    
    # RSI (14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop index (14)
    atr_period = 14
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0], low[0], close[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = np.where((max_high - min_low) > 0, 
                    100 * np.log10(atr * atr_period / (max_high - min_low)) / np.log10(atr_period), 
                    50)
    
    # Align daily indicators to higher timeframe (we're on 1d, so no alignment needed)
    kama_1d_aligned = kama_1d
    rsi_aligned = rsi
    chop_aligned = chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        kama_up = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_down = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        # RSI conditions
        rsi_over_50 = rsi_aligned[i] > 50
        rsi_under_50 = rsi_aligned[i] < 50
        
        # Chop conditions (range-bound market)
        chop_high = chop_aligned[i] > 61.8  # range
        chop_low = chop_aligned[i] < 38.2   # trend
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + Chop > 61.8 (range)
            if kama_up and rsi_over_50 and chop_high:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + Chop > 61.8 (range)
            elif kama_down and rsi_under_50 and chop_high:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down or Chop < 38.2 (trend)
            if kama_down or chop_low:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up or Chop < 38.2 (trend)
            if kama_up or chop_low:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0