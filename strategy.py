#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation
- Donchian(20) breakout captures medium-term momentum with controlled frequency
- 1d HMA21 defines trend: only long when price > HMA21, short when price < HMA21
- Volume confirmation (> 1.5x 20-period MA) reduces false breakouts
- Designed for 4h timeframe targeting 20-50 trades/year to minimize fee drag
- Uses HMA for smoother trend identification vs EMA/SMA
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(arr, weights_half, mode='valid') / weights_half.sum()
    wma_half = np.concatenate([np.full(half_period - 1, np.nan), wma_half])
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(arr, weights_full, mode='valid') / weights_full.sum()
    wma_full = np.concatenate([np.full(period - 1, np.nan), wma_full])
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    hma = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    hma = np.concatenate([np.full(sqrt_period - 1, np.nan), hma])
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: upper = max(high, 20), lower = min(low, 20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1d HMA21 for trend filter
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21, 20)  # need Donchian, HMA, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND above 1d HMA21 AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below 1d HMA21 AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian level OR crosses 1d HMA21
            exit_signal = False
            if position == 1:
                # Exit long when price < Donchian lower OR < 1d HMA21
                if close[i] < donchian_lower_aligned[i] or close[i] < hma_21_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > Donchian upper OR > 1d HMA21
                if close[i] > donchian_upper_aligned[i] or close[i] > hma_21_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dHMA21_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0