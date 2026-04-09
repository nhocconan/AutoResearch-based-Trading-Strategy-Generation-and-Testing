#!/usr/bin/env python3
# 6h_donchian_weekly_pivot_volume_v1
# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Enters long when price breaks above Donchian upper band with volume spike and weekly pivot > previous weekly pivot (bullish bias).
# Enters short when price breaks below Donchian lower band with volume spike and weekly pivot < previous weekly pivot (bearish bias).
# Uses discrete sizing (±0.25) to minimize fee churn. Designed for low trade frequency (target: 50-150 total trades over 4 years).
# Works in bull/bear by using weekly pivot bias as regime filter - only trades in direction of weekly momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h HTF data for Donchian calculation (same as primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channels (20-period)
    upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (completed 6h candle only)
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    
    # 1w HTF data for weekly pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (high + low + close) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate weekly pivot change (current vs previous) for bias
    weekly_pivot_prev = np.roll(weekly_pivot_aligned, 1)
    weekly_pivot_prev[0] = np.nan  # First value has no previous
    weekly_pivot_rising = weekly_pivot_aligned > weekly_pivot_prev  # Bullish bias
    weekly_pivot_falling = weekly_pivot_aligned < weekly_pivot_prev  # Bearish bias
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_pivot_prev[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower band
            if close[i] < lower_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper band
            if close[i] > upper_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above upper band with volume spike and bullish weekly bias
            if (close[i] > upper_20_aligned[i]) and \
               (vol_spike[i]) and \
               (weekly_pivot_rising[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band with volume spike and bearish weekly bias
            elif (close[i] < lower_20_aligned[i]) and \
                 (vol_spike[i]) and \
                 (weekly_pivot_falling[i]):
                position = -1
                signals[i] = -0.25
    
    return signals