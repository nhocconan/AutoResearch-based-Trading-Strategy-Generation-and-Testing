#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) AND 1d HMA21 rising AND volume > 1.5x 20-period MA.
Short when price breaks below lower Donchian(20) AND 1d HMA21 falling AND volume > 1.5x 20-period MA.
Exit when price touches opposite Donchian band or 1d HMA21 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades. Discrete sizing 0.25 to minimize fee churn.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Calculate 4h Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Calculate 1d HMA21 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    hma_21_1d = np.full_like(close_1d, np.nan)
    if len(wma_half) >= half_len and len(wma_full) >= 21:
        diff = 2 * wma_half[half_len-1:] - wma_full
        hma_21_1d[20:] = wma(diff, sqrt_len)
    
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate volume MA (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21, 20)  # Donchian, HMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        up = upper[i]
        low_band = lower[i]
        hma_val = hma_21_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate HMA21 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            hma_prev = hma_21_aligned[i-1]
            hma_rising = hma_val > hma_prev
            hma_falling = hma_val < hma_prev
        else:
            hma_rising = False
            hma_falling = False
        
        # Volume filter: volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above upper Donchian AND HMA21 rising AND volume filter
            if price > up and hma_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian AND HMA21 falling AND volume filter
            elif price < low_band and hma_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian OR HMA21 starts falling
                if price < low_band or (i >= start_idx + 1 and hma_val < hma_21_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian OR HMA21 starts rising
                if price > up or (i >= start_idx + 1 and hma_val > hma_21_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dHMA21_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0