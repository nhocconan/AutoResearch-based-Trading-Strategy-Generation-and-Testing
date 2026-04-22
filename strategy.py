#!/usr/bin/env python3
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
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend direction
    close_12h = df_12h['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    n = len(close_12h)
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    wma_half = wma(close_12h, half_n)
    wma_full = wma(close_12h, 21)
    wma_2xhalf = 2 * wma_half
    
    # Align lengths for subtraction
    min_len = min(len(wma_2xhalf), len(wma_full))
    diff = wma_2xhalf[-min_len:] - wma_full[-min_len:]
    hma_12h = wma(diff, sqrt_n)
    
    # Pad to match original length
    hma_12h_full = np.full(n, np.nan)
    hma_12h_full[-len(hma_12h):] = hma_12h
    
    # Align 12h HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_full)
    
    # Load 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian(20) channels on 4h data
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 4h timeframe (no additional delay needed)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian(20) AND price > 12h HMA (uptrend) with volume
            if (close[i] > upper_20_aligned[i] and 
                close[i] > hma_12h_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) AND price < 12h HMA (downtrend) with volume
            elif (close[i] < lower_20_aligned[i] and 
                  close[i] < hma_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel or 12h HMA
            if position == 1:
                if close[i] < lower_20_aligned[i] or close[i] < hma_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i] or close[i] > hma_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_12hHMA21_Trend_Volume_Session"
timeframe = "4h"
leverage = 1.0