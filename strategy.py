#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation
# Long when: price breaks above Donchian upper channel AND 1d HMA(21) rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower channel AND 1d HMA(21) falling AND volume > 1.5x 20-period MA
# Exit when: price crosses opposite Donchian channel OR HMA trend reverses
# Uses Donchian for breakout structure, HMA for trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d for HMA. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian_1dHMA_VolumeConfirm"
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
    
    # Calculate Donchian(20) channels
    if len(high) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
    
    # Calculate HMA(21) on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(data, window):
            if len(data) < window:
                return np.full_like(data, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights/weights.sum(), mode='valid')
        
        n = 21
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        
        wma_half = wma(close_1d, half_n)
        wma_full = wma(close_1d, n)
        wma_2xhalf = 2 * wma_half
        
        # Align arrays: need to handle different lengths from convolution
        raw = wma_2xhalf[-len(wma_full):] - wma_full
        hma_1d = wma(raw, sqrt_n)
        
        # Pad to original length
        hma_1d_padded = np.full(len(close_1d), np.nan)
        hma_1d_padded[half_n + sqrt_n - 1:] = hma_1d
        hma_1d = hma_1d_padded
    else:
        hma_1d = np.full(len(df_1d), np.nan)
    
    # HMA trend: rising if current > previous, falling if current < previous
    hma_rising = np.zeros(len(df_1d), dtype=bool)
    hma_falling = np.zeros(len(df_1d), dtype=bool)
    for i in range(1, len(hma_1d)):
        if not np.isnan(hma_1d[i]) and not np.isnan(hma_1d[i-1]):
            if hma_1d[i] > hma_1d[i-1]:
                hma_rising[i] = True
            elif hma_1d[i] < hma_1d[i-1]:
                hma_falling[i] = True
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Align 1d HMA trends to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above upper channel + HMA rising + volume filter
            if (close[i] > donch_high[i-1] and  # breakout above previous upper channel
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below lower channel + HMA falling + volume filter
            elif (close[i] < donch_low[i-1] and  # breakout below previous lower channel
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower channel OR HMA trend turns falling
            if (close[i] < donch_low[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper channel OR HMA trend turns rising
            if (close[i] > donch_high[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals