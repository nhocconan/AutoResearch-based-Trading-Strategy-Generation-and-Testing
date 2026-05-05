#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend filter + volume confirmation
# Long when: close > Donchian upper(20) AND 1w HMA sloping up AND volume > 1.5x 20-period MA
# Short when: close < Donchian lower(20) AND 1w HMA sloping down AND volume > 1.5x 20-period MA
# Exit when: price crosses opposite Donchian band OR volume < 1.2x 20-period MA
# Uses Donchian for structure, 1w HMA for higher-timeframe trend, volume for conviction
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
    
    # Donchian channels on 1d
    lookback = 20
    if len(high) >= lookback:
        # Upper band: highest high over lookback period
        highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        # Lower band: lowest low over lookback period
        lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_exit = volume < (1.2 * vol_ma_20)  # weaker filter for exit
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_exit = np.ones(n, dtype=bool)  # allow exit if insufficient volume data
    
    # Get 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need sufficient data for HMA
        return np.zeros(n)
    
    # Calculate HMA(21) on 1w close
    close_1w = df_1w['close'].values
    if len(close_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(data, window):
            weights = np.arange(1, window + 1)
            return np.convolve(data, weights, mode='valid') / weights.sum()
        
        # Pad arrays for convolution
        wma_half = np.full_like(close_1w, np.nan)
        wma_full = np.full_like(close_1w, np.nan)
        
        if len(close_1w) >= half_len:
            wma_values_half = wma(close_1w, half_len)
            wma_half[half_len-1:] = wma_values_half
        
        if len(close_1w) >= 21:
            wma_values_full = wma(close_1w, 21)
            wma_full[20:] = wma_values_full
        
        # HMA = WMA(2*WMA(half) - WMA(full))
        wma_diff = 2 * wma_half - wma_full
        hma_1w = np.full_like(close_1w, np.nan)
        
        if len(wma_diff) >= sqrt_len:
            hma_values = wma(wma_diff[~np.isnan(wma_diff)], sqrt_len)
            # Find valid start index
            valid_start = np.where(~np.isnan(wma_diff))[0]
            if len(valid_start) >= sqrt_len:
                hma_1w[valid_start[sqrt_len-1]:] = hma_values
    else:
        hma_1w = np.full(len(df_1w), np.nan)
    
    # HMA slope: rising if current > previous, falling if current < previous
    hma_slope = np.diff(hma_1w, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # Align 1w HMA slope to 1d timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(hma_rising_aligned[i]) or 
            np.isnan(hma_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band + rising HMA + volume filter
            if (close[i] > highest_high[i] and 
                hma_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + falling HMA + volume filter
            elif (close[i] < lowest_low[i] and 
                  hma_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band OR volume weakens
            if (close[i] < lowest_low[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band OR volume weakens
            if (close[i] > highest_high[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals