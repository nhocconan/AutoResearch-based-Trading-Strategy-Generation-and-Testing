#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and 1d volume spike confirmation.
Long when price breaks above Donchian upper (20) AND 1d HMA21 is rising AND 1d volume > 2.0x 20-period average.
Short when price breaks below Donchian lower (20) AND 1d HMA21 is falling AND 1d volume > 2.0x 20-period average.
Exit when price touches the opposite Donchian level or 1d HMA21 reverses direction.
Uses 1d HTF for trend and volume filters to reduce whipsaws and false breakouts.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels from 20-period high/low. HMA reduces lag vs EMA/SMA.
"""

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
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(arr, weights_half, mode='valid') / weights_half.sum()
    wma_half = np.pad(wma_half, (half_period - 1, 0), mode='edge')
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(arr, weights_full, mode='valid') / weights_full.sum()
    wma_full = np.pad(wma_full, (period - 1, 0), mode='edge')
    
    # Raw HMA = 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    hma = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    hma = np.pad(hma, (sqrt_period - 1, 0), mode='edge')
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d HMA21 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 1d Donchian channels (20) for entry/exit
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper = max(high, 20)
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower = min(low, 20)
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Calculate 1d volume average for spike filter (HTF)
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 20)  # Donchian (20), HMA (21), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_21_aligned[i]) or np.isnan(upper_1d_aligned[i]) or 
            np.isnan(lower_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        hma_val = hma_21_aligned[i]
        upper = upper_1d_aligned[i]
        lower = lower_1d_aligned[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        
        # Calculate HMA21 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            hma_prev = hma_21_aligned[i-1]
            hma_rising = hma_val > hma_prev
            hma_falling = hma_val < hma_prev
        else:
            hma_rising = False
            hma_falling = False
        
        if position == 0:
            # Long: Break above Donchian upper AND HMA21 rising AND volume spike
            if price > upper and hma_rising and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND HMA21 falling AND volume spike
            elif price < lower and hma_falling and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches lower Donchian OR HMA21 starts falling
                if price < lower or (i >= start_idx + 1 and hma_val < hma_21_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches upper Donchian OR HMA21 starts rising
                if price > upper or (i >= start_idx + 1 and hma_val > hma_21_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dHMA21_Trend_1dVolumeSpike"
timeframe = "4h"
leverage = 1.0