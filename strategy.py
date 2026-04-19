#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for volume and trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d volume spike (volume > 2.0 * 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 2.0)
    
    # Calculate 1w trend (close > 50-period SMA)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    trend_1w = close_1w > sma_50_1w
    
    # Calculate 4h Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(vol_spike_1d_aligned[i]) or np.isnan(trend_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Breakout conditions
        long_breakout = high[i] > high_max[i-1]
        short_breakout = low[i] < low_min[i-1]
        
        if position == 0:
            # Long when bullish 1w trend + volume spike + upward breakout
            if trend_1w_aligned[i] and vol_spike_1d_aligned[i] and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short when bearish 1w trend + volume spike + downward breakout
            elif not trend_1w_aligned[i] and vol_spike_1d_aligned[i] and short_breakout:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when trend turns bearish or downward breakout
            if not trend_1w_aligned[i] or short_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when trend turns bullish or upward breakout
            if trend_1w_aligned[i] or long_breakout:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals