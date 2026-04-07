#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Filter
Long when price breaks above 20-day Donchian high with above-average volume and weekly uptrend
Short when price breaks below 20-day Donchian low with above-average volume and weekly downtrend
Exit when price crosses opposite Donchian boundary or weekly trend flips
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Donchian Channels (20-period) ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().values
    donchian_low = low_roll.min().values
    
    # === Weekly Trend (HMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    hma_21 = calculate_hma(close_1w, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # === Volume Filter ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian low OR weekly trend turns down
            if close[i] < donchian_low[i] or hma_21_aligned[i] < np.nan_to_num(hma_21_aligned[i-1], nan=0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian high OR weekly trend turns up
            if close[i] > donchian_high[i] or hma_21_aligned[i] > np.nan_to_num(hma_21_aligned[i-1], nan=0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need expanding volume (above average)
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with volume confirmation and weekly trend alignment
            if close[i] > donchian_high[i] and hma_21_aligned[i] > np.nan_to_num(hma_21_aligned[i-1], nan=0):
                # Price above Donchian high with rising weekly trend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < donchian_low[i] and hma_21_aligned[i] < np.nan_to_num(hma_21_aligned[i-1], nan=0):
                # Price below Donchian low with falling weekly trend -> short
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
    for i in range(half_period - 1, len(arr)):
        start = i - half_period + 1
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.dot(arr[start:i+1], weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        start = i - period + 1
        weights = np.arange(1, period + 1)
        wma_full[i] = np.dot(arr[start:i+1], weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw_hma with sqrt_period
    hma = np.full_like(arr, np.nan)
    for i in range(sqrt_period - 1, len(arr)):
        start = i - sqrt_period + 1
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.dot(raw_hma[start:i+1], weights) / weights.sum()
    
    return hma