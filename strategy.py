#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_VolumeConfirm_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 12h high/low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: 20-period high
    upper_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    lower_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate volume ratio (current volume / 20-period average)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_12h > 0, volume / vol_ma_12h, 1.0)
    
    # Align all indicators to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: require volume > 1.5x average
        volume_filter = vol_ratio_aligned[i] > 1.5
        
        if position == 0:
            # Long when price breaks above upper Donchian band AND above 1d EMA50
            if (close[i] > upper_aligned[i] and 
                close[i] > ema50_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian band AND below 1d EMA50
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema50_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below lower Donchian band
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above upper Donchian band
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals