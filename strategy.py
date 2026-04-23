#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + HMA21 trend filter + volume spike confirmation.
- Primary timeframe: 4h, HTF: 1d for volume average calculation (more stable)
- Donchian breakout: price > highest(high,20) for long, < lowest(low,20) for short
- Trend filter: price > HMA21(close) for long bias, < HMA21 for short bias (using 1d HMA aligned)
- Volume confirmation: volume > 2.0 x 20-period average (strong spike filter to reduce trades)
- Exit: Donchian breakout in opposite direction or volume drops below average
- Uses proven winning pattern: price channel structure + volume confirmation + trend filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.30 to balance return and fee drag
- Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with confirmation)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan, dtype=float)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / (weights.sum() * np.ones_like(values))
    
    wma_half = pd.Series(arr).rolling(window=half, min_periods=half).apply(
        lambda x: np.dot(x, np.arange(1, half+1)) / (half*(half+1)/2), raw=True).values
    wma_full = pd.Series(arr).rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, np.arange(1, period+1)) / (period*(period+1)/2), raw=True).values
    
    raw_hma = 2 * wma_half - wma_full
    hma = pd.Series(raw_hma).rolling(window=sqrt, min_periods=sqrt).apply(
        lambda x: np.dot(x, np.arange(1, sqrt+1)) / (sqrt*(sqrt+1)/2), raw=True).values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 20-period average (strong spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate HMA21 for trend filter
    hma_21 = calculate_hma(close, 21)
    
    # Load 1d data ONCE before loop for HMA trend (more stable HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA21 for trend filter
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21)  # Need 20 for Donchian/volume, 21 for HMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(hma_21[i]) or 
            np.isnan(hma_21_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian channels (20-period)
        highest_20 = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        lowest_20 = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price > Donchian high + price > HMA21 (trend) + volume spike
            if (not np.isnan(highest_20) and 
                close[i] > highest_20 and 
                close[i] > hma_21[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.30
                position = 1
            # Short: price < Donchian low + price < HMA21 (trend) + volume spike
            elif (not np.isnan(lowest_20) and 
                  close[i] < lowest_20 and 
                  close[i] < hma_21[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price < Donchian low OR volume drops below average
            if (not np.isnan(lowest_20) and close[i] < lowest_20) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price > Donchian high OR volume drops below average
            if (not np.isnan(highest_20) and close[i] > highest_20) or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_HMA21_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0