#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and 1d trend filter
# Designed for low trade frequency (target 20-50/year) with clear breakout logic
# Works in both bull (breakout above upper band in uptrend) and bear (breakdown below lower band in downtrend) markets
# Uses Donchian channels from 4h, volume spike from 12h for confirmation, and EMA from 1d for trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_4h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_4h, low_min_20)
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size as fraction of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(vol_avg_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above upper Donchian band + uptrend + volume spike
        if (high[i] > high_max_20_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume[i] > 2.0 * vol_avg_12h_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below lower Donchian band + downtrend + volume spike
        elif (low[i] < low_min_20_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume[i] > 2.0 * vol_avg_12h_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price returns to middle of Donchian channel
        elif position == 1 and close[i] < (high_max_20_aligned[i] + low_min_20_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (high_max_20_aligned[i] + low_min_20_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hVolume_1dEMA_Trend"
timeframe = "4h"
leverage = 1.0