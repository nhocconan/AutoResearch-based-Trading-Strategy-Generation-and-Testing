#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
# Long when: price breaks above Donchian upper(20) AND 1d HMA(21) rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Donchian lower(20) AND 1d HMA(21) falling AND volume > 1.5x 20-period MA
# Exit when: price returns to Donchian middle (mean of upper/lower) OR volume drops below average
# Uses Donchian for structure, 1d HMA for higher-timeframe trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d for HMA. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_1dHMA_VolumeConfirm"
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
    
    # Donchian(20) on 4h
    lookback = 20
    if len(high) >= lookback:
        # Rolling max/min using pandas for efficiency
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        upper = high_series.rolling(window=lookback, min_periods=lookback).max().values
        lower = low_series.rolling(window=lookback, min_periods=lookback).min().values
        middle = (upper + lower) / 2.0
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        middle = np.full(n, np.nan)
    
    # Breakout conditions
    breakout_up = close > upper  # price breaks above upper band
    breakout_down = close < lower  # price breaks below lower band
    return_to_middle = np.abs(close - middle) < 0.5 * np.abs(upper - lower)  # close to middle
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_series = pd.Series(volume)
        vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    close_1d = df_1d['close'].values
    length = 21
    if len(close_1d) >= length:
        def wma(data, window):
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights, mode='valid') / weights.sum()
        
        half_length = length // 2
        sqrt_length = int(np.sqrt(length))
        
        wma_half = wma(close_1d, half_length)
        wma_full = wma(close_1d, length)
        wma_2xhalf_minus_full = 2 * wma_half - wma_full
        hma = wma(wma_2xhalf_minus_full, sqrt_length)
        
        # Pad hma to match original length
        hma_padded = np.full(len(close_1d), np.nan)
        start_idx = length - 1  # WMA reduces length by window-1
        end_idx = start_idx + len(hma)
        hma_padded[start_idx:end_idx] = hma
        
        # HMA direction: rising if current > previous, falling if current < previous
        hma_rising = np.zeros(len(close_1d), dtype=bool)
        hma_falling = np.zeros(len(close_1d), dtype=bool)
        for i in range(1, len(hma_padded)):
            if not np.isnan(hma_padded[i]) and not np.isnan(hma_padded[i-1]):
                if hma_padded[i] > hma_padded[i-1]:
                    hma_rising[i] = True
                elif hma_padded[i] < hma_padded[i-1]:
                    hma_falling[i] = True
    else:
        hma_rising = np.full(len(df_1d), False)
        hma_falling = np.full(len(df_1d), False)
        hma_padded = np.full(len(df_1d), np.nan)
    
    # Align 1d HMA direction to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or 
            np.isnan(return_to_middle[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout up + 1d HMA rising + volume filter
            if (breakout_up[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout down + 1d HMA falling + volume filter
            elif (breakout_down[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to middle OR volume drops below average
            if (return_to_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to middle OR volume drops below average
            if (return_to_middle[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals