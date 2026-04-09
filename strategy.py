#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA(34) trend + volume confirmation + chop regime filter
# Donchian captures breakouts in both directions; 1d HMA confirms higher timeframe trend
# Volume ensures breakout authenticity; chop regime avoids whipsaws in ranging markets
# Discrete sizing 0.25 limits drawdown; target 75-200 total trades over 4 years

name = "4h_1d_donchian_hma_volume_chop_v1"
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
    
    # Load 1d data ONCE before loop for HMA and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA(34)
    close_1d = df_1d['close'].values
    half_len = 34 // 2
    sqrt_len = int(np.sqrt(34))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        wma_vals = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            wma_vals[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
        return wma_vals
    
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 34)
    hma_1d = 2 * wma_half - wma_full
    hma_1d = wma(hma_1d, sqrt_len)
    
    # Align 1d HMA to 4h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d chop regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # True Range
    tr = np.full(len(close_1d_arr), np.nan)
    for i in range(1, len(close_1d_arr)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d_arr[i-1]),
            abs(low_1d[i] - close_1d_arr[i-1])
        )
    tr[0] = high_1d[0] - low_1d[0]
    
    # ATR
    atr_1d = np.full(len(close_1d_arr), np.nan)
    for i in range(13, len(close_1d_arr)):  # min_periods=14
        atr_1d[i] = np.mean(tr[i-13:i+1])
    
    # Sum of absolute price changes
    abs_changes = np.abs(np.diff(close_1d_arr, prepend=close_1d_arr[0]))
    sum_abs_changes = np.full(len(close_1d_arr), np.nan)
    for i in range(13, len(close_1d_arr)):
        sum_abs_changes[i] = np.sum(abs_changes[i-13:i+1])
    
    # Chop = 100 * log10(sum_abs_changes/atr) / log10(14)
    chop_1d = np.full(len(close_1d_arr), np.nan)
    for i in range(13, len(close_1d_arr)):
        if sum_abs_changes[i] > 0 and atr_1d[i] > 0:
            chop_1d[i] = 100 * np.log10(sum_abs_changes[i] / atr_1d[i]) / np.log10(14)
        else:
            chop_1d[i] = 50.0  # neutral
    
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        # Chop regime filter: avoid extreme ranging (chop > 61.8) or extreme trending (chop < 38.2)
        chop_filter = (chop_1d_aligned[i] >= 38.2) and (chop_1d_aligned[i] <= 61.8)
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < 1d HMA (trend change)
            if close[i] < donchian_low[i] or close[i] < hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > 1d HMA (trend change)
            if close[i] > donchian_high[i] or close[i] > hma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, chop filter, and Donchian breakout + 1d HMA filter
            if volume_confirmed and chop_filter:
                # Long entry: price > Donchian high AND price > 1d HMA (bullish alignment)
                if close[i] > donchian_high[i] and close[i] > hma_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < 1d HMA (bearish alignment)
                elif close[i] < donchian_low[i] and close[i] < hma_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals