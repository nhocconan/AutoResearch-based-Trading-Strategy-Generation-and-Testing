#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend filter + volume confirmation
# Long when: price > upper Donchian(20) AND 12h HMA(21) rising AND volume > 1.5x 20-period MA
# Short when: price < lower Donchian(20) AND 12h HMA(21) falling AND volume > 1.5x 20-period MA
# Exit when: price crosses middle Donchian(20) (mean reversion) OR volume drops below average
# Uses Donchian for breakout structure, HMA for trend direction, volume for conviction
# Timeframe: 4h, HTF: 12h. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_12hHMA_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Get 12h data ONCE before loop for HMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 12h
    close_12h = df_12h['close'].values
    if len(close_12h) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(data, window):
            if len(data) < window:
                return np.full_like(data, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
            hma = wma(raw_hma, sqrt_len)
            # Pad to match original length
            hma_padded = np.full(len(close_12h), np.nan)
            hma_padded[half_len:half_len+len(hma)] = hma
            hma_12h = hma_padded
        else:
            hma_12h = np.full(len(close_12h), np.nan)
    else:
        hma_12h = np.full(len(close_12h), np.nan)
    
    # HMA rising/falling
    hma_rising = np.zeros(len(hma_12h), dtype=bool)
    hma_falling = np.zeros(len(hma_12h), dtype=bool)
    for i in range(1, len(hma_12h)):
        if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
            hma_rising[i] = hma_12h[i] > hma_12h[i-1]
            hma_falling[i] = hma_12h[i] < hma_12h[i-1]
    
    # Align 12h HMA to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_12h, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_12h, hma_falling.astype(float))
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(hma_rising_aligned[i]) or 
            np.isnan(hma_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper Donchian + HMA rising + volume filter
            if (close[i] > donchian_upper[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower Donchian + HMA falling + volume filter
            elif (close[i] < donchian_lower[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below middle Donchian OR volume drops
            if (close[i] < donchian_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above middle Donchian OR volume drops
            if (close[i] > donchian_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals