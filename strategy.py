#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend and volume confirmation
# Designed for low trade frequency (target 20-40/year) with clear trend-following logic
# Uses 4h price channel breakouts, confirmed by 12h EMA trend and volume spikes
# Works in trending markets (both bull and bear) by following the 12h trend direction
# Avoids choppy markets through volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA50 on 12h for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 4h)
    vol_avg = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_4h, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_20)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_avg_aligned = align_htf_to_ltf(prices, df_4h, vol_avg)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size as fraction of capital
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian upper band + uptrend + volume spike
        if (high[i] > highest_20_aligned[i] and 
            close[i] > ema50_12h_aligned[i] and 
            volume[i] > 2.0 * vol_avg_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian lower band + downtrend + volume spike
        elif (low[i] < lowest_20_aligned[i] and 
              close[i] < ema50_12h_aligned[i] and 
              volume[i] > 2.0 * vol_avg_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price returns to middle of channel
        elif position == 1 and close[i] < (highest_20_aligned[i] + lowest_20_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > (highest_20_aligned[i] + lowest_20_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hEMA_Volume_Trend"
timeframe = "4h"
leverage = 1.0