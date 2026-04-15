#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w EMA trend filter
# Targets breakouts in trending markets while filtering sideways action with EMA trend
# Works in bull markets (breakouts above upper band in uptrend) and bear markets (breakdowns below lower band in downtrend)
# Uses 20-period Donchian channels, 1d volume spike for confirmation, and 50-period 1w EMA for trend alignment
# Designed for low trade frequency (target 20-40/year) with clear trend-following logic

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    # Using rolling window with min_periods to avoid look-ahead
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period on 1d)
    vol_avg = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1w for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    high_max_20_aligned = align_htf_to_ltf(prices, df_4h, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_4h, low_min_20)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size as fraction of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(vol_avg_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: price breaks above upper Donchian band + uptrend + volume spike
        if (high[i] > high_max_20_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below lower Donchian band + downtrend + volume spike
        elif (low[i] < low_min_20_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
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

name = "4h_Donchian_1dVolume_1wEMA_Trend"
timeframe = "4h"
leverage = 1.0