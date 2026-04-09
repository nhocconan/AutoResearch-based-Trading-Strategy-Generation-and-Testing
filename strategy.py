#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and HMA trend filter
# Donchian breakout captures strong momentum moves in both bull and bear markets
# Volume confirmation ensures breakout authenticity (avoids false breakouts)
# HMA(21) trend filter: only take breakouts in direction of higher timeframe trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_donchian_hma_volume_v1"
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
    
    # Load 1d data ONCE before loop for volume and HMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d HMA(21) - Hull Moving Average
    def wma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights/weights.sum(), mode='valid')
    
    def hma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = wma(values, half_period)
        wma_full = wma(values, period)
        hma_values = 2 * wma_half - wma_full
        hma_values = wma(hma_values, sqrt_period)
        # Pad with NaN to match original length
        result = np.full(len(values), np.nan)
        result[period-1:] = hma_values
        return result
    
    hma_21_1d = hma(df_1d['close'].values, 21)
    
    # Align 1d indicators to 4h timeframe (wait for 1d bar close)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(avg_volume_1d_aligned[i]) or np.isnan(hma_21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 1d average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        # Trend filter: price above/below 1d HMA(21)
        uptrend = close[i] > hma_21_1d_aligned[i]
        downtrend = close[i] < hma_21_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band OR trend reverses
            if close[i] < lowest_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band OR trend reverses
            if close[i] > highest_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: breakout with volume confirmation and trend alignment
            if close[i] > highest_high[i] and volume_confirmed and uptrend:
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and volume_confirmed and downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals