#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter and volume spike (>1.5x 20-bar avg) for confirmation.
# Uses 1d HMA21 for trend alignment (HTF), 4h Donchian channels for breakout entry, and volume confirmation to avoid false breakouts.
# Designed for moderate trade frequency (target 75-200 total over 4 years) to minimize fee drag while capturing strong trends.
# Works in both bull and bear markets by following the 1d trend direction and requiring volume confirmation.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(arr)
    for i in range(half_period, len(arr)):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.dot(arr[i - half_period + 1:i + 1], weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.zeros_like(arr)
    for i in range(period, len(arr)):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.dot(arr[i - period + 1:i + 1], weights) / weights.sum()
    
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    
    # WMA of raw HMA with sqrt period
    hma = np.zeros_like(arr)
    for i in range(sqrt_period, len(arr)):
        if not np.isnan(raw_hma[i]):
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.dot(raw_hma[i - sqrt_period + 1:i + 1], weights) / weights.sum()
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d HMA21 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Donchian channels (20-period) for breakout (primary TF)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel, close > 1d HMA21, volume spike (>1.5x avg)
            if (high[i] > highest_high[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower channel, close < 1d HMA21, volume spike (>1.5x avg)
            elif (low[i] < lowest_low[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Donchian lower channel or volume drops
            if (low[i] < lowest_low[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Donchian upper channel or volume drops
            if (high[i] > highest_high[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals