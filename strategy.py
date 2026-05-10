#!/usr/bin/env python3
# 12h_1d_KAMA_Direction_With_Volume_And_Chop_Filter
# Hypothesis: Use 1d Kaufman Adaptive Moving Average (KAMA) direction for trend bias on 12h timeframe.
# Enter long when price crosses above KAMA with volume confirmation and chop regime (range-bound).
# Enter short when price crosses below KAMA with volume confirmation and chop regime.
# Exit on opposite KAMA cross or chop regime breakdown (trending market).
# Designed to work in both bull and bear markets by adapting to volatility and avoiding whipsaws in strong trends.
# Target: 15-25 trades/year (~60-100 total over 4 years) to stay within optimal trade frequency for 12h.

name = "12h_1d_KAMA_Direction_With_Volume_And_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smooth ER
    er_smooth = pd.Series(er).ewm(alpha=1/er_length, adjust=False).mean().values
    # Smoothing constants
    sc = (er_smooth * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))**2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d KAMA trend filter (ER length 10, fast 2, slow 30)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    kama_1d = calculate_kama(df_1d['close'].values)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Chop filter: Choppiness Index > 61.8 (range-bound regime)
    # Calculate Chop: 100 * log10(sum(ATR, 14) / (max(high,14) - min(low,14))) / log10(14)
    atr_list = []
    for i in range(len(high)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    atr = np.array(atr_list)
    
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    # Avoid division by zero
    ratio = np.where(range_hl != 0, atr_sum / range_hl, 1)
    chop = 100 * np.log10(ratio) / np.log10(14)
    chop_filter = chop > 61.8  # Range-bound regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above KAMA with volume confirmation in choppy market
            if (close[i] > kama_1d_aligned[i] and 
                close[i-1] <= kama_1d_aligned[i-1] and 
                volume_filter[i] and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with volume confirmation in choppy market
            elif (close[i] < kama_1d_aligned[i] and 
                  close[i-1] >= kama_1d_aligned[i-1] and 
                  volume_filter[i] and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA OR chop regime breaks (trending market)
            if (close[i] < kama_1d_aligned[i] or 
                chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA OR chop regime breaks (trending market)
            if (close[i] > kama_1d_aligned[i] or 
                chop[i] <= 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals