#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour trend filter (HMA) and volume confirmation.
Long when price breaks above Donchian upper band, 12h HMA trending up, and volume > 1.3x average.
Short when price breaks below Donchian lower band, 12h HMA trending down, and volume > 1.3x average.
Exit when price returns to Donchian middle band or trend reverses.
Designed for moderate trade frequency (~30-50/year) to capture breakouts in trending markets.
Works in bull markets via upward breakouts and bear markets via downward breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_channel = (highest_high + lowest_low) / 2
    
    # Load 12-hour data for HMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour HMA (21-period)
    close_12h = df_12h['close'].values
    # HMA formula: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights/weights.sum(), mode='valid')
    
    # Pad arrays for WMA calculation
    wma_half = np.full_like(close_12h, np.nan)
    wma_full = np.full_like(close_12h, np.nan)
    
    for i in range(half_n, len(close_12h)):
        wma_half[i] = np.dot(close_12h[i-half_n+1:i+1], np.arange(1, half_n+1)) / (half_n*(half_n+1)/2)
    
    for i in range(n_hma, len(close_12h)):
        wma_full[i] = np.dot(close_12h[i-n_hma+1:i+1], np.arange(1, n_hma+1)) / (n_hma*(n_hma+1)/2)
    
    # HMA = WMA(2*WMA(half) - WMA(full))
    wma_diff = 2 * wma_half - wma_full
    hma_12h = np.full_like(close_12h, np.nan)
    for i in range(sqrt_n-1, len(wma_diff)):
        if not np.isnan(wma_diff[i]):
            start_idx = i - sqrt_n + 1
            hma_12h[i] = np.dot(wma_diff[start_idx:i+1], np.arange(1, sqrt_n+1)) / (sqrt_n*(sqrt_n+1)/2)
    
    # HMA slope for trend direction
    hma_slope = np.diff(hma_12h, prepend=np.nan)
    
    # Load volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_slope[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band, HMA trending up, volume confirmation
            if (close[i] > highest_high[i] and hma_slope[i] > 0 and 
                volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, HMA trending down, volume confirmation
            elif (close[i] < lowest_low[i] and hma_slope[i] < 0 and 
                  volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle band OR HMA trend turns down
                if close[i] <= middle_channel[i] or hma_slope[i] < 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle band OR HMA trend turns up
                if close[i] >= middle_channel[i] or hma_slope[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_12hHMA_Volume_Breakout"
timeframe = "4h"
leverage = 1.0