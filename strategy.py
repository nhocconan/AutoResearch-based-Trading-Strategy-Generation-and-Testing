#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Breakout with Weekly Pivot Direction and Volume Confirmation
# Uses Donchian channel breakout (20-period) from 6h for entry signals
# Weekly pivot points provide directional bias (price above weekly pivot = long bias, below = short bias)
# Volume confirmation (>1.8x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of weekly trend
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivots to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate Donchian Channel (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for Donchian Channel
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r2_1w_aligned[i]) or 
            np.isnan(s2_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Weekly pivot bias: price above pivot = long bias, below = short bias
        long_bias = price > pivot_1w_aligned[i]
        short_bias = price < pivot_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume filter and long bias
            if price > upper_channel[i] and vol > 1.8 * avg_vol[i] and long_bias:
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower channel with volume filter and short bias
            elif price < lower_channel[i] and vol > 1.8 * avg_vol[i] and short_bias:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below pivot (trend change) or lower channel
            if price < pivot_1w_aligned[i] or price < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above pivot (trend change) or upper channel
            if price > pivot_1w_aligned[i] or price > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0