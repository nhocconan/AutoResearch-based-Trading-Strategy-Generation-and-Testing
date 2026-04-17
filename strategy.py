#!/usr/bin/env python3
"""
4h_4H_1D_Volume_Donchian_Trend_Filter
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian high 
with daily volume confirmation and daily trend alignment; short when breaks below 
Donchian low with volume and opposite daily trend. Daily trend uses 50-period SMA to 
avoid counter-trend trades. Designed for 20-50 total trades over 4 years to minimize 
fee drag and work in both bull/bear markets via trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for trend and volume ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily 50-period SMA for trend
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Daily 20-period average volume for confirmation
    vol_avg20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    # === 4h data for Donchian channels (20-period) ===
    # Use 4h high/low for Donchian calculation
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Warmup: covers 50-day SMA and 20-day volume average
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(sma50_1d_aligned[i]) or np.isnan(vol_avg20_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.5x 20-day average
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        # Trend filter: price above/below daily 50 SMA
        above_trend = close[i] > sma50_1d_aligned[i]
        below_trend = close[i] < sma50_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price > 4h Donchian high + volume + above daily trend
            if close[i] > donchian_high[i] and vol_filter and above_trend:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < 4h Donchian low + volume + below daily trend
            elif close[i] < donchian_low[i] and vol_filter and below_trend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal at opposite Donchian level
        elif position == 1:
            if close[i] < donchian_low[i]:  # break below Donchian low = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > donchian_high[i]:  # break above Donchian high = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4H_1D_Volume_Donchian_Trend_Filter"
timeframe = "4h"
leverage = 1.0