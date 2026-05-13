#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses Donchian channel breakouts for momentum capture, HMA21 for smooth 1d trend alignment,
# and volume filter to reduce false signals. Designed for low trade frequency (20-50/year) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by HMA).
# Exit on reverse Donchian touch or volume drop below 60% of average.

name = "4h_Donchian20_1dHMA21_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1d HMA21 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_1d).ewm(span=half, adjust=False).mean().values
    wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate Donchian channels (20-period) based on prior bars
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel, close > 1d HMA21, volume spike (>1.8x avg)
            if (high[i] > highest_high[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower channel, close < 1d HMA21, volume spike (>1.8x avg)
            elif (low[i] < lowest_low[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Donchian lower channel or volume drops
            if (low[i] < lowest_low[i]) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Donchian upper channel or volume drops
            if (high[i] > highest_high[i]) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals