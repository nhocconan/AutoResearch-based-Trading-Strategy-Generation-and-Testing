#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d daily pivot direction filter
# - Long when price breaks above Donchian high (20-bar high) AND daily pivot is bullish (close > pivot)
# - Short when price breaks below Donchian low (20-bar low) AND daily pivot is bearish (close < pivot)
# - Exit when price crosses back through the Donchian midpoint
# - Uses volume filter: require volume > 1.5x 20-period average to avoid false breakouts
# - Designed for low trade frequency (target 15-25/year) with clear trend-following logic
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard: P = (H+L+C)/3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Pivot bias: bullish if close > pivot, bearish if close < pivot
    pivot_bias = np.where(close_1d > pivot_1d, 1, np.where(close_1d < pivot_1d, -1, 0))
    
    # Donchian channels (20-period)
    lookback = 20
    # Calculate rolling max/min manually for efficiency
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20-1, n):
        vol_ma[i] = np.mean(volume[i-20+1:i+1])
    volume_filter = volume > (1.5 * vol_ma)
    
    # Align 1d pivot bias to 6h timeframe
    pivot_bias_aligned = align_htf_to_ltf(prices, df_1d, pivot_bias)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(pivot_bias_aligned[i]) or np.isnan(volume_filter[i])):
            continue
        
        # Long entry: price breaks above Donchian high AND bullish pivot bias AND volume filter
        if (close[i] > donchian_high[i] and 
            pivot_bias_aligned[i] > 0 and 
            volume_filter[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian low AND bearish pivot bias AND volume filter
        elif (close[i] < donchian_low[i] and 
              pivot_bias_aligned[i] < 0 and 
              volume_filter[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: price crosses back through Donchian midpoint
        elif position != 0:
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if position == 1 and close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > midpoint:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Pivot_Bias_Volume"
timeframe = "6h"
leverage = 1.0