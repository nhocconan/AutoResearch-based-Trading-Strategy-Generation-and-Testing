#!/usr/bin/env python3
"""
4h_Donchian20_HMA21_VolumeFilter_V2
Strategy: 4h Donchian(20) breakout with HMA(21) trend filter and volume confirmation.
Long: Close breaks above 20-period high AND HMA rising AND volume > 1.5x average.
Short: Close breaks below 20-period low AND HMA falling AND volume > 1.5x average.
Exit: Opposite breakout OR trend reversal.
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Works in bull/bear via HMA trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(array, period):
    """Hull Moving Average"""
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(arr, window):
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights, 'valid') / weights.sum()
    
    if len(array) < period:
        return np.full_like(array, np.nan)
    
    wma_half = wma(array, half_period)
    wma_full = wma(array, period)
    
    # Handle array lengths
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad to original length
    result = np.full_like(array, np.nan)
    start_idx = period - half_period  # offset due to WMAs
    end_idx = start_idx + len(hma)
    if end_idx <= len(array):
        result[start_idx:end_idx] = hma
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and HMA
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 20-day high and low (Donchian channels)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # HMA(21) on daily close
    hma_21 = calculate_hma(close_1d, 21)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for HMA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions: HMA rising/falling
        hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
        hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Donchian breakout conditions
        breakout_long = close[i] > high_20_aligned[i]
        breakout_short = close[i] < low_20_aligned[i]
        
        if position == 0:
            # Long: breakout above 20-day high + HMA rising + volume
            if breakout_long and hma_rising and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below 20-day low + HMA falling + volume
            elif breakout_short and hma_falling and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below 20-day low OR HMA falling OR loss of volume
            if breakout_short or not hma_rising or not vol_confirm:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above 20-day high OR HMA rising OR loss of volume
            if breakout_long or hma_rising or not vol_confirm:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_HMA21_VolumeFilter_V2"
timeframe = "4h"
leverage = 1.0