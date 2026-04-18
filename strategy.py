#!/usr/bin/env python3
"""
12h_KAMA_Direction_Volume_Regime
KAMA direction signal on 12h with volume confirmation and 1d chop filter:
- Long when KAMA slope > 0 + volume > 1.5x 20-period average + chop > 61.8 (range)
- Short when KAMA slope < 0 + volume > 1.5x 20-period average + chop > 61.8 (range)
- Exit when KAMA slope changes sign
- Designed for 12-30 trades/year per symbol
Works in choppy markets (range-bound conditions) with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    if n < er_length:
        return kama
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.concatenate([np.full(er_length-1, np.nan), er])
    
    # Smoothing Constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama[er_length-1] = close[er_length-1]
    for i in range(er_length, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def calculate_chop(high, low, close, period=14):
    """Calculate Choppiness Index."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period-1, n):
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else 0,
                     abs(low[j] - close[j-1]) if j > 0 else 0)
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high != lowest_low:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate KAMA on 12h data
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    
    # KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=np.nan)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need 10 for ER + 20 for vol MA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_slope[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Chop filter: chop > 61.8 indicates ranging market
        chop_ok = chop_1d_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA slope up + volume + chop
            if kama_slope[i] > 0 and volume_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: KAMA slope down + volume + chop
            elif kama_slope[i] < 0 and volume_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA slope turns down
            if kama_slope[i] < 0:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA slope turns up
            if kama_slope[i] > 0:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_Volume_Regime"
timeframe = "12h"
leverage = 1.0