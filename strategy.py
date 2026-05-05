#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
# Long when: price > Donchian upper(20) AND close > HMA(21) on 1d AND volume > 1.5x 20-period MA
# Short when: price < Donchian lower(20) AND close < HMA(21) on 1d AND volume > 1.5x 20-period MA
# Exit when: price crosses Donchian midpoint OR trend reverses
# Uses Donchian for structure, HMA for trend filter, volume for conviction
# Timeframe: 4h, HTF: 1d for HMA. Target: 75-200 total trades over 4 years (19-50/year).

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
    
    # Calculate Donchian channels on 4h
    lookback = 20
    if len(high) >= lookback:
        highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        donchian_mid = (highest_high + lowest_low) / 2
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate HMA(21) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 21:
        half_length = 21 // 2
        sqrt_length = int(np.sqrt(21))
        
        # WMA function
        def wma(data, period):
            if len(data) < period:
                return np.full_like(data, np.nan)
            weights = np.arange(1, period + 1)
            result = np.convolve(data, weights, mode='valid') / weights.sum()
            padded = np.full_like(data, np.nan)
            padded[period-1:] = result
            return padded
        
        wma_half = wma(close_1d, half_length)
        wma_full = wma(close_1d, 21)
        hma_raw = 2 * wma_half - wma_full
        hma_21 = wma(hma_raw, sqrt_length)
    else:
        hma_21 = np.full(len(df_1d), np.nan)
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Align 1d HMA to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(hma_21_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above upper band + uptrend + volume
            if (close[i] > highest_high[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below lower band + downtrend + volume
            elif (close[i] < lowest_low[i] and 
                  close[i] < hma_21_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses midpoint OR trend reverses
            if (close[i] < donchian_mid[i] or close[i] < hma_21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses midpoint OR trend reverses
            if (close[i] > donchian_mid[i] or close[i] > hma_21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals