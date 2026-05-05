#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d trend filter (HMA21) + volume confirmation
# Long when: price breaks above Donchian upper (20) AND 1d HMA21 rising AND volume > 1.5x 24-period MA
# Short when: price breaks below Donchian lower (20) AND 1d HMA21 falling AND volume > 1.5x 24-period MA
# Exit when: price returns to Donchian midpoint OR trend reverses
# Uses Donchian for structure, 1d HMA for major trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.
# BTC/ETH focus: trend filter helps avoid whipsaws in ranging markets, volume confirms institutional participation.

name = "4h_Donchian20_1dHMA21_VolumeConfirm"
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
    
    # Calculate volume confirmation on 4h using 24-period MA (equivalent to 1d lookback)
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
    
    # Calculate 21-period HMA on 1d timeframe
    if len(close_1d) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA helper
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMA for close
        wma_close = np.full_like(close_1d, np.nan)
        for i in range(len(close_1d)):
            if i >= 20:  # need 21 points for WMA(21)
                wma_close[i] = wma(close_1d[i-20:i+1], 21)
        
        # WMA(half_len)
        wma_half = np.full_like(close_1d, np.nan)
        for i in range(len(close_1d)):
            if i >= half_len - 1:
                wma_half[i] = wma(close_1d[i-half_len+1:i+1], half_len)
        
        # 2*WMA(half) - WMA(full)
        diff = 2 * wma_half - wma_close
        
        # WMA of diff with sqrt_len
        hma_1d = np.full_like(close_1d, np.nan)
        for i in range(len(close_1d)):
            if i >= sqrt_len - 1 and not np.isnan(diff[i]):
                hma_1d[i] = wma(diff[i-sqrt_len+1:i+1], sqrt_len)
        
        # HMA slope (rising/falling)
        hma_slope = np.diff(hma_1d, prepend=np.nan)
        hma_rising = hma_slope > 0
        hma_falling = hma_slope < 0
    else:
        hma_rising = np.full(len(close_1d), False)
        hma_falling = np.full(len(close_1d), False)
    
    # Align 1d HMA trend to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    # Calculate Donchian channels on 4h (20-period)
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
            # Long conditions: price breaks above Donchian upper + 1d HMA rising + volume filter
            if (close[i] > donch_high[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + 1d HMA falling + volume filter
            elif (close[i] < donch_low[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR 1d HMA turns falling
            if (close[i] <= donch_mid[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR 1d HMA turns rising
            if (close[i] >= donch_mid[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals