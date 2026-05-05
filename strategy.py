#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation
# Long when: price breaks above Donchian upper band AND 1d HMA(21) is rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower band AND 1d HMA(21) is falling AND volume > 1.5x 20-period MA
# Exit when: price reverts to Donchian midpoint OR volume drops below average
# Uses Donchian for price channel breakouts, 1d HMA for multi-timeframe trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d for HMA. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian_1dHMA_VolumeConfirm_v2"
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
    
    # Calculate Donchian(20) on 4h
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Breakout conditions
    breakout_up = close > donchian_upper  # price breaks above upper band
    breakout_down = close < donchian_lower  # price breaks below lower band
    revert_to_mid = np.abs(close - donchian_mid) < (donchian_upper - donchian_lower) * 0.1  # near midpoint
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_exit = volume < vol_ma_20  # exit when volume drops below average
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_exit = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(data, window):
        if len(data) < window:
            return np.full_like(data, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    if len(close_1d) >= 21:
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        # Pad to align arrays
        wma_half_padded = np.concatenate([np.full(half_len - 1, np.nan), wma_half])
        wma_full_padded = np.concatenate([np.full(20, np.nan), wma_full])  # 21-1=20
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma = wma(raw_hma, sqrt_len)
        # Pad HMA to original length
        hma_final = np.concatenate([np.full(len(close_1d) - len(hma), np.nan), hma])
    else:
        hma_final = np.full(len(close_1d), np.nan)
    
    # HMA slope: rising if current > previous, falling if current < previous
    hma_rising = np.zeros(len(hma_final), dtype=bool)
    hma_falling = np.zeros(len(hma_final), dtype=bool)
    hma_rising[1:] = hma_final[1:] > hma_final[:-1]
    hma_falling[1:] = hma_final[1:] < hma_final[:-1]
    
    # Align 1d HMA to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(revert_to_mid[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(volume_exit[i]) or np.isnan(hma_rising_aligned[i]) or 
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
            # Exit long: revert to midpoint OR volume exit OR HMA falls
            if (revert_to_mid[i] or volume_exit[i] or hma_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: revert to midpoint OR volume exit OR HMA rises
            if (revert_to_mid[i] or volume_exit[i] or hma_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals