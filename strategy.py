#!/usr/bin/env python3
# 12h_KAMA_Direction_With_Volume_and_Chop_Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction. In choppy markets (CHOP>61.8), avoid trades.
# In trending markets (CHOP<=61.8), go long when price > KAMA(14) with volume > 1.5x average, short when price < KAMA(14).
# Volume confirmation filters false signals. Designed to work in both bull and bear markets by following the trend.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_KAMA_Direction_With_Volume_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Fix: volatility should be rolling sum of absolute changes
    volatility = pd.Series(change).rolling(window=er_length, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_chop(high, low, close, window=14):
    """Calculate Choppiness Index"""
    atr = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=window, min_periods=window).mean().sum()
    # Fix: True Range calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum()
    max_high = pd.Series(high).rolling(window=window, min_periods=window).max()
    min_low = pd.Series(low).rolling(window=window, min_periods=window).min()
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(window)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Chop on daily timeframe
    chop = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate KAMA on 12h timeframe
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (CHOP <= 61.8)
        if chop_aligned[i] > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA with volume confirmation
            if close[i] > kama[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA with volume confirmation
            elif close[i] < kama[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price < KAMA (trend change)
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price > KAMA (trend change)
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals