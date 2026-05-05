#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
# Long when: price breaks above Donchian(20) high + 1d HMA(21) upward + volume > 1.5x 24-period MA
# Short when: price breaks below Donchian(20) low + 1d HMA(21) downward + volume > 1.5x 24-period MA
# Exit when: price crosses Donchian(20) midpoint OR volume filter fails
# Uses Donchian for structure, HMA for trend filter, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_1dHMA21_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) on 12h
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate volume confirmation on 12h using 24-period MA (equivalent to 1d lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on 1d: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA for close_1d
    wma_full = np.full(len(close_1d), np.nan)
    wma_half = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 21:
        wma_full[20:] = wma(close_1d, 21)
    if len(close_1d) >= half_n:
        wma_half[half_n-1:] = wma(close_1d, half_n)
    
    # HMA = WMA(2*WMA(half_n) - WMA(n)), sqrt_n
    wma_diff = 2 * wma_half - wma_full
    hma_1d = np.full(len(close_1d), np.nan)
    if len(wma_diff) >= sqrt_n:
        hma_1d[sqrt_n-1:] = wma(wma_diff[sqrt_n-1:], sqrt_n)
    
    # Align HMA to 12h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate HMA slope for trend direction
    hma_slope = np.full(len(hma_1d_aligned), np.nan)
    for i in range(1, len(hma_1d_aligned)):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]):
            hma_slope[i] = hma_1d_aligned[i] - hma_1d_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(hma_slope[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian high + HMA upward + volume filter
            if (close[i] > donchian_high[i] and 
                hma_slope[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian low + HMA downward + volume filter
            elif (close[i] < donchian_low[i] and 
                  hma_slope[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian mid OR HMA turns down OR volume filter fails
            if (close[i] < donchian_mid[i] or 
                hma_slope[i] <= 0 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian mid OR HMA turns up OR volume filter fails
            if (close[i] > donchian_mid[i] or 
                hma_slope[i] >= 0 or 
                not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals