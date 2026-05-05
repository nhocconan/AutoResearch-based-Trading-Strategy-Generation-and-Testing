#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation
# Long when: price breaks above Donchian upper band (20-period high) + 1d HMA(21) rising + volume > 1.5x 24-period MA
# Short when: price breaks below Donchian lower band (20-period low) + 1d HMA(21) falling + volume > 1.5x 24-period MA
# Exit when: price returns to Donchian middle (10-period average of upper/lower) or HMA trend reverses
# Uses Donchian for structure, 1d HMA for higher-timeframe trend, volume for conviction
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
    
    # Calculate Donchian channels on 12h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
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
    
    # Calculate 1d HMA(21) - Hull Moving Average
    if len(close_1d) >= 21:
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA(21/2) of close
        wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
        ).values
        
        # WMA(21) of close
        wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
        ).values
        
        # HMA = 2*WMA(half) - WMA(full)
        hma_raw = 2 * wma_half - wma_full
        # Final WMA(sqrt(21)) of the above
        hma_21 = pd.Series(hma_raw).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
            lambda x: np.average(x, weights=np.arange(1, len(x)+1)), raw=True
        ).values
        
        # Calculate HMA slope (rising/falling)
        hma_slope = np.diff(hma_21, prepend=hma_21[0])
        hma_rising = hma_slope > 0
        hma_falling = hma_slope < 0
    else:
        hma_21 = np.full(len(close_1d), np.nan)
        hma_rising = np.zeros(len(close_1d), dtype=bool)
        hma_falling = np.zeros(len(close_1d), dtype=bool)
    
    # Align 1d HMA trends to 12h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper + 1d HMA rising + volume filter
            if (close[i] > donchian_upper[i] and 
                hma_rising_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower + 1d HMA falling + volume filter
            elif (close[i] < donchian_lower[i] and 
                  hma_falling_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian middle OR 1d HMA turns falling
            if (close[i] < donchian_middle[i] or not hma_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian middle OR 1d HMA turns rising
            if (close[i] > donchian_middle[i] or not hma_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals