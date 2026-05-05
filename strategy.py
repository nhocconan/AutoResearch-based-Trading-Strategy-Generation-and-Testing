#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter (HMA21) + volume confirmation
# Long when: price breaks above Donchian upper (20) AND 1w HMA21 rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower (20) AND 1w HMA21 falling AND volume > 1.5x 20-period MA
# Exit when: price returns to Donchian midpoint OR trend reverses
# Uses Donchian for structure, 1w HMA for major trend filter, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wHMA21_VolumeConfirm"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA helper
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMA for close
        wma_close = np.full_like(close_1w, np.nan)
        for i in range(len(close_1w)):
            if i >= 20:  # need 21 points for WMA(21)
                wma_close[i] = wma(close_1w[i-20:i+1], 21)
        
        # WMA(half_len)
        wma_half = np.full_like(close_1w, np.nan)
        for i in range(len(close_1w)):
            if i >= half_len - 1:
                wma_half[i] = wma(close_1w[i-half_len+1:i+1], half_len)
        
        # 2*WMA(half) - WMA(full)
        diff = 2 * wma_half - wma_close
        
        # WMA of diff with sqrt_len
        hma_1w = np.full_like(close_1w, np.nan)
        for i in range(len(close_1w)):
            if i >= sqrt_len - 1 and not np.isnan(diff[i]):
                hma_1w[i] = wma(diff[i-sqrt_len+1:i+1], sqrt_len)
        
        # HMA slope (rising/falling)
        hma_slope = np.diff(hma_1w, prepend=np.nan)
        hma_rising = hma_slope > 0
        hma_falling = hma_slope < 0
    else:
        hma_rising = np.full(len(close_1w), False)
        hma_falling = np.full(len(close_1w), False)
    
    # Align 1w HMA trend to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling.astype(float))
    
    # Calculate Donchian channels on 1d (20-period)
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
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + 1w HMA rising + volume filter
            if (close[i] > donch_high[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + 1w HMA falling + volume filter
            elif (close[i] < donch_low[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR 1w HMA turns falling
            if (close[i] <= donch_mid[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR 1w HMA turns rising
            if (close[i] >= donch_mid[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals