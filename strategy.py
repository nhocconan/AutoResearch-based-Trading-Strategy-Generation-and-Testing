#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout + 1d HMA trend + volume confirmation
# Long when: price breaks above Donchian(20) upper + 1d HMA(21) upward + volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower + 1d HMA(21) downward + volume > 1.5x 20-period MA
# Exit when: price crosses Donchian midpoint OR volume drops below average
# Uses Donchian for structure, 1d HMA for HTF trend filter, volume for conviction
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_Breakout_1dHMA21_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 12h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Calculate volume confirmation on 12h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate HMA(21) on 1d timeframe
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    if len(close_1d) >= 21:
        # WMA helper function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMAs
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        
        # 2*WMA(n/2) - WMA(n)
        diff = 2 * wma_half - wma_full
        
        # WMA of the diff with sqrt(n) period
        if len(diff) >= sqrt_len:
            hma_values = wma(diff, sqrt_len)
            # Pad to match original length
            hma_1d = np.full(len(close_1d), np.nan)
            hma_1d[half_len - 1:half_len - 1 + len(hma_values)] = hma_values
        else:
            hma_1d = np.full(len(close_1d), np.nan)
    else:
        hma_1d = np.full(len(close_1d), np.nan)
    
    # Calculate HMA slope for trend direction
    hma_slope = np.diff(hma_1d, prepend=np.nan)
    
    # Align 1d HMA and slope to 12h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(hma_1d_aligned[i]) or 
            np.isnan(hma_slope_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + HMA upward + volume filter
            if (close[i] > donchian_upper[i] and 
                hma_slope_aligned[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + HMA downward + volume filter
            elif (close[i] < donchian_lower[i] and 
                  hma_slope_aligned[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR volume drops below average
            if (close[i] < donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR volume drops below average
            if (close[i] > donchian_mid[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals