#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
# Long when: price breaks above Donchian(20) upper band AND 1w HMA(21) rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian(20) lower band AND 1w HMA(21) falling AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian(20) midpoint OR volume drops below average
# Uses Donchian for structure, HMA for smooth trend filter, volume for conviction
# Timeframe: 1d, HTF: 1w for HMA. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Donchian20_1wHMA_VolumeConfirm"
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
    
    # Calculate Donchian(20) on 1d
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Donchian breakout signals
    breakout_up = close > donchian_upper
    breakout_down = close < donchian_lower
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(data, window):
            if len(data) < window:
                return np.full_like(data, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        hma_raw = 2 * wma_half - wma_full
        hma = wma(hma_raw, sqrt_len)
        
        # Pad to original length
        hma_padded = np.full(len(close_1w), np.nan)
        hma_padded[half_len:] = hma[:len(close_1w)-half_len]
        hma_1w = hma_padded
        
        # HMA slope: rising if current > previous, falling if current < previous
        hma_slope = np.diff(hma_1w, prepend=np.nan)
        hma_rising = hma_slope > 0
        hma_falling = hma_slope < 0
    else:
        hma_rising = np.full(len(df_1w), np.nan)
        hma_falling = np.full(len(df_1w), np.nan)
    
    # Align 1w HMA to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(hma_rising_aligned[i]) or 
            np.isnan(hma_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout up + HMA rising + volume filter
            if (breakout_up[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout down + HMA falling + volume filter
            elif (breakout_down[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: breakout down OR HMA falling OR volume drops
            if (breakout_down[i] or hma_falling_aligned[i] == 1.0 or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: breakout up OR HMA rising OR volume drops
            if (breakout_up[i] or hma_rising_aligned[i] == 1.0 or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals