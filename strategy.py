#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolumeTrend_4hTrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume trend and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4-day average volume for trend filter
    volume_4d_avg = np.full(len(volume_1d), np.nan)
    for i in range(4, len(volume_1d)):
        volume_4d_avg[i] = np.mean(volume_1d[i-4:i])
    
    volume_trend_up = volume_1d > volume_4d_avg
    
    # Calculate 20-period Donchian channels on 1d data
    upper_channel = np.full(len(high_1d), np.nan)
    lower_channel = np.full(len(low_1d), np.nan)
    
    for i in range(19, len(high_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 4h EMA for trend filter
    close_series = pd.Series(close)
    ema_4h = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    vol_trend_aligned = align_htf_to_ltf(prices, df_1d, volume_trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_4h[i]) or np.isnan(vol_trend_aligned[i])):
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long when price breaks above upper Donchian with volume trend and above EMA
            if close[i] > upper_aligned[i] and vol_trend_aligned[i] and close[i] > ema_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian with volume trend and below EMA
            elif close[i] < lower_aligned[i] and vol_trend_aligned[i] and close[i] < ema_4h[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below lower Donchian
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above upper Donchian
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals