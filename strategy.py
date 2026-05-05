#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend filter + volume confirmation
# Long when: price breaks above Donchian(20) high AND 1w HMA(21) rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) low AND 1w HMA(21) falling AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian(20) midpoint OR volume drops below average
# Uses Donchian for breakout structure, 1w HMA for higher-timeframe trend, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wHMA_VolumeConfirm"
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
    
    # Calculate Donchian(20) channels on 1d
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1w close
    close_1w = df_1w['close'].values
    if len(close_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_n = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = np.array([np.nan] * half_n + [wma(close_1w[i:i+half_n], half_n) 
                          for i in range(len(close_1w) - half_n + 1)])
        wma_full = np.array([np.nan] * 21 + [wma(close_1w[i:i+21], 21) 
                          for i in range(len(close_1w) - 21 + 1)])
        hma_1w = 2 * wma_half - wma_full
        hma_1w = np.array([np.nan] * (21 - sqrt_n) + [wma(hma_1w[i:i+sqrt_n], sqrt_n) 
                          for i in range(len(hma_1w) - sqrt_n + 1)])
        
        # HMA rising/falling
        hma_rising = np.zeros(len(hma_1w), dtype=bool)
        hma_falling = np.zeros(len(hma_1w), dtype=bool)
        for i in range(1, len(hma_1w)):
            if not np.isnan(hma_1w[i]) and not np.isnan(hma_1w[i-1]):
                hma_rising[i] = hma_1w[i] > hma_1w[i-1]
                hma_falling[i] = hma_1w[i] < hma_1w[i-1]
    else:
        hma_1w = np.full(len(close_1w), np.nan)
        hma_rising = np.zeros(len(close_1w), dtype=bool)
        hma_falling = np.zeros(len(close_1w), dtype=bool)
    
    # Align 1w HMA signals to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling.astype(float))
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_normal = volume > (0.5 * vol_ma_20)  # for exit condition
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_normal = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(hma_rising_aligned[i]) or 
            np.isnan(hma_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above Donchian high + rising 1w HMA + volume filter
            if (close[i] > donchian_high[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + falling 1w HMA + volume filter
            elif (close[i] < donchian_low[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price below Donchian mid OR volume drops below normal
            if (close[i] < donchian_mid[i] or volume_normal[i] == 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above Donchian mid OR volume drops below normal
            if (close[i] > donchian_mid[i] or volume_normal[i] == 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals