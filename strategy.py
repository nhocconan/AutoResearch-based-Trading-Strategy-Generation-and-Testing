#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA21 trend + volume confirmation
# Long when: price breaks above Donchian(20) upper band AND 1w HMA21 rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND 1w HMA21 falling AND volume > 1.5x 20-period MA
# Exit when: price returns to Donchian(20) middle band OR trend reverses
# Uses Donchian for structure, 1w HMA for trend filter, volume for conviction
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
    
    # Calculate Donchian channels on 1d (20-period)
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = highest_high
        donchian_lower = lowest_low
        donchian_middle = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_middle = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = close > donchian_upper
    donchian_breakout_down = close < donchian_lower
    donchian_reentry_middle = (close > donchian_middle) & (close < donchian_upper) | \
                              (close < donchian_middle) & (close > donchian_lower)
    
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
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Calculate WMA for half length
        wma_half = np.full_like(close_1w, np.nan)
        for i in range(half_len - 1, len(close_1w)):
            wma_half[i] = np.mean(close_1w[i - half_len + 1:i + 1] * np.arange(1, half_len + 1))
        
        # Calculate WMA for full length
        wma_full = np.full_like(close_1w, np.nan)
        for i in range(21 - 1, len(close_1w)):
            wma_full[i] = np.mean(close_1w[i - 21 + 1:i + 1] * np.arange(1, 22))
        
        # Calculate raw HMA: 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # Calculate final HMA: WMA(raw_hma, sqrt_len)
        hma_21 = np.full_like(close_1w, np.nan)
        for i in range(sqrt_len - 1, len(raw_hma)):
            if not np.isnan(raw_hma[i - sqrt_len + 1:i + 1]).any():
                hma_21[i] = np.mean(raw_hma[i - sqrt_len + 1:i + 1] * np.arange(1, sqrt_len + 1))
        
        # Trend direction: rising if current > previous
        hma_rising = np.diff(hma_21, prepend=np.nan) > 0
        hma_falling = np.diff(hma_21, prepend=np.nan) < 0
    else:
        hma_rising = np.full(len(close_1w), False)
        hma_falling = np.full(len(close_1w), False)
    
    # Align 1w HMA trend to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + 1w HMA rising + volume filter
            if (donchian_breakout_up[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + 1w HMA falling + volume filter
            elif (donchian_breakout_down[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian middle OR 1w HMA turns falling
            if (donchian_reentry_middle[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian middle OR 1w HMA turns rising
            if (donchian_reentry_middle[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals