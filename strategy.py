#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation
# Long when: price breaks above Donchian(20) upper band AND price > 1d HMA21 (uptrend) AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND price < 1d HMA21 (downtrend) AND volume > 1.5x 20-period MA
# Exit when: price returns to Donchian(20) middle band OR trend reverses
# Uses Donchian for structure, 1d HMA for HTF trend, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_Breakout_1dHMA21_VolumeConfirm"
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
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_middle = (highest_high + lowest_low) / 2
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Calculate volume confirmation on 4h using 20-period MA
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
    
    # Calculate 21-period HMA on 1d timeframe
    if len(close_1d) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights/weights.sum(), mode='valid')
        
        # Pad arrays for WMA calculation
        wma_half = np.full_like(close_1d, np.nan)
        wma_full = np.full_like(close_1d, np.nan)
        
        if len(close_1d) >= half_len:
            wma_half[half_len-1:] = wma(close_1d, half_len)
        if len(close_1d) >= 21:
            wma_full[20:] = wma(close_1d, 21)
        
        # HMA calculation
        hma_raw = 2 * wma_half - wma_full
        hma_21 = np.full_like(close_1d, np.nan)
        if len(hma_raw) >= sqrt_len:
            hma_21[sqrt_len-1:] = wma(hma_half[half_len-1:], sqrt_len)  # Simplified: use WMA of raw HMA
        
        # Alternative simpler HMA approximation using EMA for stability
        # HMA ~ EMA of (2*EMA(n/2) - EMA(n))
        ema_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
        ema_full = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
        hma_approx = 2 * ema_half - ema_full
        hma_21 = pd.Series(hma_approx).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
        
        hma_rising = np.diff(hma_21, prepend=np.nan) > 0
        hma_falling = np.diff(hma_21, prepend=np.nan) < 0
    else:
        hma_rising = np.full(len(close_1d), False)
        hma_falling = np.full(len(close_1d), False)
    
    # Align 1d HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(hma_rising_aligned[i]) or 
            np.isnan(hma_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + 1d HMA rising + volume filter
            if (close[i] > highest_high[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + 1d HMA falling + volume filter
            elif (close[i] < lowest_low[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian middle OR 1d HMA turns falling
            if (close[i] <= donchian_middle[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian middle OR 1d HMA turns rising
            if (close[i] >= donchian_middle[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals