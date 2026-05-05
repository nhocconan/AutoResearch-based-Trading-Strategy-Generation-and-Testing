#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
# Long when: price breaks above Donchian upper channel AND 1w HMA trending up AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower channel AND 1w HMA trending down AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian middle channel OR HMA trend reverses
# Uses Donchian for structure, HMA for HTF trend filter, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wHMA21_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 1d timeframe
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Hull Moving Average (HMA) on 1w timeframe
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    if len(close_1w) >= 21:
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        # WMA helper function
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMA(21)
        wma_21 = np.full(len(close_1w), np.nan)
        if len(close_1w) >= 21:
            wma_21_vals = wma(close_1w, 21)
            wma_21[20:] = wma_21_vals
        
        # Calculate WMA(10.5) -> WMA(10) for half period
        wma_half = np.full(len(close_1w), np.nan)
        if len(close_1w) >= half_n:
            wma_half_vals = wma(close_1w, half_n)
            wma_half[half_n-1:] = wma_half_vals
        
        # 2*WMA(half) - WMA(21)
        diff = 2 * wma_half - wma_21
        
        # WMA of diff with sqrt(21) period
        hma_1w = np.full(len(close_1w), np.nan)
        if len(diff) >= sqrt_n:
            # Find valid start index for diff
            valid_start = half_n - 1  # where wma_half becomes valid
            if valid_start < len(diff):
                hma_vals = wma(diff[valid_start:], sqrt_n)
                hma_1w[valid_start + sqrt_n - 1:] = hma_vals
    else:
        hma_1w = np.full(len(close_1w), np.nan)
    
    # Align 1w HMA to 1d timeframe
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(hma_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper AND HMA trending up AND volume filter
            if (close[i] > donchian_upper[i] and 
                hma_1w_aligned[i] > hma_1w_aligned[i-1] and  # HMA trending up
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower AND HMA trending down AND volume filter
            elif (close[i] < donchian_lower[i] and 
                  hma_1w_aligned[i] < hma_1w_aligned[i-1] and  # HMA trending down
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian middle OR HMA trend turns down
            if (close[i] < donchian_middle[i] or 
                hma_1w_aligned[i] < hma_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian middle OR HMA trend turns up
            if (close[i] > donchian_middle[i] or 
                hma_1w_aligned[i] > hma_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals