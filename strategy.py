#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA21 trend + volume confirmation
# Donchian breakout provides structure-based entries, 1d HMA21 ensures we trade with higher timeframe trend
# Volume confirmation filters weak breakouts (volume > 1.5x 20-period average)
# Works in bull/bear: HMA21 trend filter avoids counter-trend whipsaws during ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_donchian_hma21_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 1d HMA21 for trend direction
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < 4h Donchian Low OR price < 1d HMA21 (trend change)
            if close[i] < donchian_low[i] or close[i] < hma_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > 4h Donchian High OR price > 1d HMA21 (trend change)
            if close[i] > donchian_high[i] or close[i] > hma_21_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + HMA21 trend filter
            if volume_confirmed:
                # Long entry: price > 4h Donchian High AND price > 1d HMA21 (bullish breakout + uptrend)
                if close[i] > donchian_high[i] and close[i] > hma_21_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < 4h Donchian Low AND price < 1d HMA21 (bearish breakout + downtrend)
                elif close[i] < donchian_low[i] and close[i] < hma_21_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        if i < half_period - 1:
            wma_half[i] = np.nan
        else:
            weights = np.arange(1, half_period + 1)
            wma_half[i] = np.sum(arr[i-half_period+1:i+1] * weights) / np.sum(weights)
    
    # WMA of full period
    wma_full = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        if i < period - 1:
            wma_full[i] = np.nan
        else:
            weights = np.arange(1, period + 1)
            wma_full[i] = np.sum(arr[i-period+1:i+1] * weights) / np.sum(weights)
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw HMA with sqrt(period)
    hma = np.full_like(arr, np.nan)
    for i in range(len(arr)):
        if i < sqrt_period - 1:
            hma[i] = np.nan
        else:
            weights = np.arange(1, sqrt_period + 1)
            hma[i] = np.sum(raw_hma[i-sqrt_period+1:i+1] * weights) / np.sum(weights)
    
    return hma