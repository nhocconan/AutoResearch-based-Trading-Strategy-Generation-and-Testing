#!/usr/bin/env python3

# 1d_HMA_Filtered_Breakout
# Hypothesis: 1d price breaks above/below HMA(20) with volume confirmation and HMA(50) trend filter
# HMA(20) acts as dynamic support/resistance; breakouts indicate momentum shifts
# HMA(50) filter ensures trades align with medium-term trend, reducing whipsaws
# Volume > 1.5x 20-period average confirms institutional participation
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in bull/bear via HMA(50) trend filter - avoids counter-trend trades

name = "1d_HMA_Filtered_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=np.float64)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA helper
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan, dtype=np.float64)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMAs
    wma_half = wma(arr, half_period)
    wma_full = wma(arr, period)
    
    # Align arrays (WMA reduces length)
    diff = 2 * wma_half - wma_full
    # Pad to match original length
    pad_width = len(arr) - len(diff)
    if pad_width > 0:
        diff = np.concatenate([np.full(pad_width, np.nan), diff])
    elif pad_width < 0:
        diff = diff[-pad_width:]
    
    # Final WMA
    hma = wma(diff, sqrt_period)
    pad_width = len(arr) - len(hma)
    if pad_width > 0:
        hma = np.concatenate([np.full(pad_width, np.nan), hma])
    elif pad_width < 0:
        hma = hma[-pad_width:]
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for HMA calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for HMA(50)
        return np.zeros(n)
    
    # Calculate HMA(20) and HMA(50) on weekly data
    hma_20 = calculate_hma(df_1w['close'].values, 20)
    hma_50 = calculate_hma(df_1w['close'].values, 50)
    
    # Align to daily timeframe
    hma_20_aligned = align_htf_to_ltf(prices, df_1w, hma_20)
    hma_50_aligned = align_htf_to_ltf(prices, df_1w, hma_50)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for HMA(50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_20_aligned[i]) or np.isnan(hma_50_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above HMA(50) for long, below for short
        uptrend = close[i] > hma_50_aligned[i]
        downtrend = close[i] < hma_50_aligned[i]
        
        if position == 0:
            # Long: price crosses above HMA(20) with volume and uptrend
            if (close[i] > hma_20_aligned[i] and 
                close[i-1] <= hma_20_aligned[i-1] and  # Cross above
                volume_confirm[i] and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below HMA(20) with volume and downtrend
            elif (close[i] < hma_20_aligned[i] and 
                  close[i-1] >= hma_20_aligned[i-1] and  # Cross below
                  volume_confirm[i] and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below HMA(20) or trend changes
            if (close[i] < hma_20_aligned[i] and 
                close[i-1] >= hma_20_aligned[i-1]):  # Cross below
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above HMA(20) or trend changes
            if (close[i] > hma_20_aligned[i] and 
                close[i-1] <= hma_20_aligned[i-1]):  # Cross above
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals