#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with 1d HMA21 trend filter and volume confirmation
# Long when price breaks above 1w Donchian(20) upper band AND 1d HMA21 is rising AND 1d volume > 1.3 * avg_volume(20)
# Short when price breaks below 1w Donchian(20) lower band AND 1d HMA21 is falling AND 1d volume > 1.3 * avg_volume(20)
# Exit when price returns to 1w Donchian(20) midpoint
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides strong weekly structure, filtering out daily noise
# 1d HMA21 reduces lag vs EMA while maintaining smooth trend
# Volume confirmation filters out low-conviction breakouts
# Works in both bull (breakout continuations) and bear (breakdown continuations) markets

name = "1d_1wDonchian20_Breakout_1dHMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full_like(arr, np.nan)
    for i in range(half_period - 1, len(arr)):
        wma_half[i] = np.sum(arr[i - half_period + 1:i + 1] * np.arange(1, half_period + 1)) / (half_period * (half_period + 1) / 2)
    
    # WMA of full period
    wma_full = np.full_like(arr, np.nan)
    for i in range(period - 1, len(arr)):
        wma_full[i] = np.sum(arr[i - period + 1:i + 1] * np.arange(1, period + 1)) / (period * (period + 1) / 2)
    
    # Raw HMA = 2 * WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # WMA of raw HMA with sqrt period
    hma = np.full_like(arr, np.nan)
    for i in range(sqrt_period - 1, len(arr)):
        hma[i] = np.sum(raw_hma[i - sqrt_period + 1:i + 1] * np.arange(1, sqrt_period + 1)) / (sqrt_period * (sqrt_period + 1) / 2)
    
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian(20)
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channel (20-period)
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    mid_1w = (upper_1w + lower_1w) / 2.0
    
    # Align 1w Donchian levels to 1d timeframe (wait for completed 1w bar)
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    mid_1w_aligned = align_htf_to_ltf(prices, df_1w, mid_1w)
    
    # Get 1d data ONCE before loop for HMA21 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 completed daily bars for HMA21 and volume avg
        return np.zeros(n)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d HMA21 trend filter
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(mid_1w_aligned[i]) or np.isnan(hma_21_1d_aligned[i]) or
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper band, HMA21 rising, volume spike
            if (close[i] > upper_1w_aligned[i] and close[i-1] <= upper_1w_aligned[i-1] and 
                hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Donchian lower band, HMA21 falling, volume spike
            elif (close[i] < lower_1w_aligned[i] and close[i-1] >= lower_1w_aligned[i-1] and 
                  hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to 1w Donchian midpoint
            if close[i] <= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to 1w Donchian midpoint
            if close[i] >= mid_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals