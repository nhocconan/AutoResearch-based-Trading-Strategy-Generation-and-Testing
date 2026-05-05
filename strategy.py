#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation
# Long when: price breaks above Donchian(20) high, 1w HMA(21) trending up, volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) low, 1w HMA(21) trending down, volume > 1.5x 20-period MA
# Exit when: price retouches Donchian(20) midpoint or HMA trend reverses
# Uses Donchian for structure, 1w HMA for HTF trend, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_Breakout_1wHMA21_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 21-period HMA on 1w timeframe
    if len(close_1w) >= 21:
        # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_n)
        wma_full = wma(close_1w, 21)
        if len(wma_half) >= half_n and len(wma_full) >= 21:
            raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
            hma_21 = wma(raw_hma, sqrt_n)
            # Pad to match original length
            hma_21_padded = np.full(len(close_1w), np.nan)
            hma_21_padded[half_n + sqrt_n - 1:] = hma_21
            hma_21 = hma_21_padded
        else:
            hma_21 = np.full(len(close_1w), np.nan)
    else:
        hma_21 = np.full(len(close_1w), np.nan)
    
    # Align 1w HMA to 1d timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate Donchian(20) channels on 1d timeframe
    if len(high) >= 20 and len(low) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_high + donch_low) / 2
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + HMA trending up + volume filter
            if (close[i] > donch_high[i] and 
                hma_21_aligned[i] > hma_21_aligned[i-1] and  # HMA trending up
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + HMA trending down + volume filter
            elif (close[i] < donch_low[i] and 
                  hma_21_aligned[i] < hma_21_aligned[i-1] and  # HMA trending down
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Donchian midpoint OR HMA trend turns down
            if (close[i] <= donch_mid[i] or hma_21_aligned[i] < hma_21_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Donchian midpoint OR HMA trend turns up
            if (close[i] >= donch_mid[i] or hma_21_aligned[i] > hma_21_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals