#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation (>1.5x 20-bar avg).
# Uses 12h HMA21 for smooth trend alignment (HTF), 4h Donchian channels for breakout entry, and volume confirmation to filter weak breakouts.
# Designed for moderate trade frequency (target 75-200 total over 4 years) by requiring volume spike and trend alignment.
# Works in both bull and bear markets by following the 12h trend direction and requiring volume confirmation to avoid false signals.

name = "4h_Donchian20_Breakout_12hHMA21_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(arr)
    if n < period:
        return np.full(n, np.nan)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / (window * (window + 1) / 2)
    
    wma_half = np.full(n, np.nan)
    wma_full = np.full(n, np.nan)
    
    for i in range(half - 1, n):
        wma_half[i] = wma(arr[i - half + 1:i + 1], half)
    for i in range(period - 1, n):
        wma_full[i] = wma(arr[i - period + 1:i + 1], period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = np.full(n, np.nan)
    for i in range(sqrt_n - 1, n):
        hma[i] = wma(raw_hma[i - sqrt_n + 1:i + 1], sqrt_n)
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h HMA21 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    hma_21_12h = calculate_hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate Donchian channels (20-period) for breakout (primary TF)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(hma_21_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel, close > 12h HMA21, volume spike (>1.5x avg)
            if (high[i] > highest_high[i] and 
                close[i] > hma_21_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower channel, close < 12h HMA21, volume spike (>1.5x avg)
            elif (low[i] < lowest_low[i] and 
                  close[i] < hma_21_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Donchian lower channel or volume drops significantly
            if (low[i] < lowest_low[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Donchian upper channel or volume drops significantly
            if (high[i] > highest_high[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals