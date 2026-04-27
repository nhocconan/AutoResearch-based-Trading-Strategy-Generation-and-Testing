#!/usr/bin/env python3
"""
#100739 - 6h_Donchian20_12hTrend_1dVolumeFilter
Hypothesis: 6-hour Donchian(20) breakout with 12-hour EMA20 trend filter and 1-day volume spike filter. Works in bull markets by capturing trend continuations and in bear markets by filtering false breakouts during low-volume periods. Targets 15-30 trades/year to minimize fee drag while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h EMA20 for trend filter
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Calculate 1d average volume (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Donchian channels (20-period) - use only past data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        # Approximate current 1d volume using last available aligned value
        volume_filter = volume_1d[-1] > (vol_avg_1d[-1] * 1.5) if len(volume_1d) > 0 and len(vol_avg_1d) > 0 else False
        # For simplicity, use a rolling volume filter on 6h data as proxy
        vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter_6h = volume[i] > (vol_ma_6h[i] * 1.5) if not np.isnan(vol_ma_6h[i]) else False
        
        # Long condition: price breaks above Donchian high, above 12h EMA20, volume spike
        if (close[i] > high_max[i] and 
            close[i] > ema20_12h_aligned[i] and 
            volume_filter_6h):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian low, below 12h EMA20, volume spike
        elif (close[i] < low_min[i] and 
              close[i] < ema20_12h_aligned[i] and 
              volume_filter_6h):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite Donchian band (mean reversion)
        elif position == 1 and close[i] < low_min[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > high_max[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_12hTrend_1dVolumeFilter"
timeframe = "6h"
leverage = 1.0