#!/usr/bin/env python3
"""
6h_RangeBreakout_WeeklyTrend_Volume
Hypothesis: Uses 1d weekly trend (via 5-day SMA) to filter 6h breakouts from 20-bar Donchian channels, with volume confirmation to avoid false signals. Designed for 6h timeframe to capture medium-term trends while minimizing false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for weekly trend and Donchian context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d 5-day SMA for weekly trend filter
    close_1d = df_1d['close'].values
    sma_5_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 5:
        sma_5_1d[4] = np.mean(close_1d[:5])
        for i in range(5, len(close_1d)):
            sma_5_1d[i] = close_1d[i] * 0.2 + sma_5_1d[i-1] * 0.8  # EMA-like smoothing for simplicity
    
    # Align 1d 5-day SMA to 6h timeframe
    sma_5_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_5_1d)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    for i in range(n):
        if i < 20:
            highest_high[i] = np.max(high[:i+1])
            lowest_low[i] = np.min(low[:i+1])
        else:
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 20  # Warmup for Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(sma_5_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above 20-bar high, above 1d 5-day SMA, with volume spike
            if close[i] > highest_high[i] and close[i] > sma_5_1d_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below 20-bar low, below 1d 5-day SMA, with volume spike
            elif close[i] < lowest_low[i] and close[i] < sma_5_1d_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: reverse signal or loss of momentum
            if close[i] < lowest_low[i] or close[i] < sma_5_1d_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: reverse signal or loss of momentum
            if close[i] > highest_high[i] or close[i] > sma_5_1d_aligned[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RangeBreakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0