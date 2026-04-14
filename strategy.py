#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Works in bull markets by capturing breakouts; works in bear markets by avoiding counter-trend trades
# Uses weekly trend to filter direction, reducing false signals in choppy markets
# Volume confirmation ensures breakouts have conviction
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-week EMA for weekly trend
    if len(close_1w) < 20:
        return np.zeros(n)
    
    ema20_1w = np.full_like(close_1w, np.nan)
    alpha = 2 / (20 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema20_1w[i] = close_1w[i]
        elif np.isnan(ema20_1w[i-1]):
            ema20_1w[i] = close_1w[i]
        else:
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day Donchian channels
    high_20 = np.full_like(high_1d, np.nan)
    low_20 = np.full_like(low_1d, np.nan)
    
    for i in range(19, len(high_1d)):
        high_20[i] = np.max(high_1d[i-19:i+1])
        low_20[i] = np.min(low_1d[i-19:i+1])
    
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high + above weekly EMA + volume surge
            if (close[i] > high_20_aligned[i] and
                close[i] > ema20_1w_aligned[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 20-day low + below weekly EMA + volume surge
            elif (close[i] < low_20_aligned[i] and
                  close[i] < ema20_1w_aligned[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below 20-day low OR below weekly EMA
            if (close[i] < low_20_aligned[i] or 
                close[i] < ema20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above 20-day high OR above weekly EMA
            if (close[i] > high_20_aligned[i] or 
                close[i] > ema20_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_WeeklyEMA_Volume_Filter"
timeframe = "1d"
leverage = 1.0