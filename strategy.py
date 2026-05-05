#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation
# Long when: close breaks above Donchian upper (20) AND 1d HMA rising (trending up) AND volume > 1.5x 20-period MA
# Short when: close breaks below Donchian lower (20) AND 1d HMA falling (trending down) AND volume > 1.5x 20-period MA
# Exit when: price crosses opposite Donchian band OR volume < 1.2x 20-period MA (loss of conviction)
# Uses Donchian for structure, 1d HMA for higher-timeframe trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d for HMA. Target: 100-200 total trades over 4 years (25-50/year) to balance edge and fees.

name = "4h_Donchian_1dHMA_VolumeConfirm"
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
    
    # Donchian(20) on 4h
    lookback = 20
    if len(high) >= lookback:
        # Upper band: highest high over lookback period
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
        for i in range(lookback-1, n):
            upper[i] = np.max(high[i-lookback+1:i+1])
            lower[i] = np.min(low[i-lookback+1:i+1])
    else:
        upper = np.full(n, np.nan)
        lower = np.full(n, np.nan)
    
    # Volume confirmation on 4h
    vol_lookback = 20
    if len(volume) >= vol_lookback:
        vol_ma = pd.Series(volume).rolling(window=vol_lookback, min_periods=vol_lookback).mean().values
        volume_filter = volume > (1.5 * vol_ma)
        volume_exit = volume > (1.2 * vol_ma)  # less strict for exit
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_exit = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    close_1d = df_1d['close'].values
    length = 21
    if len(close_1d) >= length:
        def wma(data, window):
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights, 'valid') / weights.sum()
        
        half_length = length // 2
        sqrt_length = int(np.sqrt(length))
        
        wma_half = wma(close_1d, half_length)
        wma_full = wma(close_1d, length)
        wma_2xhalf_minus_full = 2 * wma_half - wma_full
        
        # Pad arrays to align with original
        wma_2xhalf_minus_full_padded = np.full(len(close_1d), np.nan)
        wma_2xhalf_minus_full_padded[half_length-1:len(wma_2xhalf_minus_full)+half_length-1] = wma_2xhalf_minus_full
        
        hma = wma(wma_2xhalf_minus_full_padded, sqrt_length)
        hma_padded = np.full(len(close_1d), np.nan)
        start_idx = sqrt_length - 1
        end_idx = start_idx + len(hma)
        hma_padded[start_idx:end_idx] = hma
        
        # HMA rising/falling: compare current to previous
        hma_rising = np.zeros(len(close_1d), dtype=bool)
        hma_falling = np.zeros(len(close_1d), dtype=bool)
        for i in range(1, len(hma_padded)):
            if not np.isnan(hma_padded[i]) and not np.isnan(hma_padded[i-1]):
                hma_rising[i] = hma_padded[i] > hma_padded[i-1]
                hma_falling[i] = hma_padded[i] < hma_padded[i-1]
    else:
        hma_rising = np.zeros(len(df_1d), dtype=bool)
        hma_falling = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d HMA signals to 4h timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(volume_exit[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper + HMA rising + volume filter
            if (close[i] > upper[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower + HMA falling + volume filter
            elif (close[i] < lower[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below lower OR volume exit (loss of conviction)
            if (close[i] < lower[i] or not volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above upper OR volume exit (loss of conviction)
            if (close[i] > upper[i] or not volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals