#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume spike confirmation
# Long when: price breaks above Donchian(20) upper band AND price > 1d HMA21 (uptrend) AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND price < 1d HMA21 (downtrend) AND volume > 1.5x 20-period MA
# Exit when: price touches Donchian(20) midpoint OR trend reverses
# Uses Donchian for structure, 1d HMA for higher-timeframe trend, volume spike for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

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
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian(20) channels on 4h
    if len(high) >= 20 and len(low) >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = highest_high
        donchian_lower = lowest_low
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    donchian_breakout_up = (close > donchian_upper) & (np.roll(close, 1) <= np.roll(donchian_upper, 1))
    donchian_breakout_down = (close < donchian_lower) & (np.roll(close, 1) >= np.roll(donchian_lower, 1))
    donchian_touch_mid = np.abs(close - donchian_mid) < (donchian_upper - donchian_lower) * 0.05  # within 5% of mid
    
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
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21_1d = wma(wma_diff, sqrt_len)
        
        # Pad to original length
        hma_21_1d_padded = np.full(len(close_1d), np.nan)
        hma_21_1d_padded[half_len:len(wma_half)+half_len] = wma_half
        hma_21_1d_padded[len(wma_half)+half_len:len(wma_half)+half_len+len(wma_full)] = wma_full
        hma_21_1d_padded[len(wma_half)+half_len+len(wma_full):len(wma_half)+half_len+len(wma_full)+len(hma_21_1d)] = hma_21_1d
        
        # Simpler approach: use EMA approximation for HMA (commonly used)
        ema_half = pd.Series(close_1d).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
        ema_full = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
        hma_approx = pd.Series(2 * ema_half - ema_full).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
        
        # Use the approximation for simplicity and speed
        hma_21_1d = hma_approx
        hma_rising = np.diff(hma_21_1d, prepend=np.nan) > 0
        hma_falling = np.diff(hma_21_1d, prepend=np.nan) < 0
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(hma_rising_aligned[i]) or 
            np.isnan(hma_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Donchian breakout up + 1d HMA rising + volume filter
            if (donchian_breakout_up[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Donchian breakout down + 1d HMA falling + volume filter
            elif (donchian_breakout_down[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian midpoint OR 1d HMA turns falling
            if (donchian_touch_mid[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian midpoint OR 1d HMA turns rising
            if (donchian_touch_mid[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals