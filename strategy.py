#!/usr/bin/env python3
"""
12h_KAMA_Direction_With_Volume_and_Chop_Filter
Hypothesis: KAUFMAN ADAPTIVE MOVING AVERAGE (KAMA) identifies trend direction
with less whipsaw than traditional MAs. Combined with volume confirmation and
Choppiness Index regime filter to avoid false signals in sideways markets.
Designed for 12h timeframe to target 50-150 total trades over 4 years.
Works in bull/bear via adaptive trend detection and regime filtering.
"""

name = "12h_KAMA_Direction_With_Volume_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(close - np.roll(close, er_length))
    change[0] = np.abs(close[0] - close[0])  # First element
    
    # Volatility sum of absolute changes
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        if i < er_length:
            volatility[i] = np.sum(np.abs(np.diff(close[:i+1])))
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR (sum of TR over period)
    atr_sum = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            atr_sum[i] = np.sum(tr[:i+1])
        else:
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(len(close)):
        if i < period:
            hh[i] = np.max(high[:i+1])
            ll[i] = np.min(low[:i+1])
        else:
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
    
    # Choppiness Index
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if atr_sum[i] > 0 and hh[i] != ll[i]:
            log_val = np.log10(atr_sum[i] / (hh[i] - ll[i]))
            chop[i] = 100 * log_val / np.log10(period)
        else:
            chop[i] = 50  # Neutral when undefined
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Choppiness Index (regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # 12h data for KAMA (trend direction)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    kama_12h = calculate_kama(df_12h['close'].values, er_length=10, fast_sc=2, slow_sc=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(kama_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when CHOP < 61.8 (trending, not choppy)
        trending_regime = chop_1d_aligned[i] < 61.8
        
        if position == 0:
            # Long: price above KAMA with volume and trending regime
            if (close[i] > kama_12h_aligned[i] and 
                volume_confirm[i] and 
                trending_regime):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume and trending regime
            elif (close[i] < kama_12h_aligned[i] and 
                  volume_confirm[i] and 
                  trending_regime):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or regime becomes choppy
            if (close[i] < kama_12h_aligned[i]) or (chop_1d_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or regime becomes choppy
            if (close[i] > kama_12h_aligned[i]) or (chop_1d_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals