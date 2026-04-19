#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolume_1wTrend_v2"
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = volume_1d / volume_ma_1d  # Current volume / 20-day average
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 50-period EMA on weekly close
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d volume ratio and 1w EMA to 4h timeframe
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 4h (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ratio_aligned[i]) or np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume_ratio_aligned[i] > 1.5
        
        # Trend filter: price above/below 50-period weekly EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long when price breaks above upper Donchian + volume + uptrend
            if close[i] > highest_high[i] and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian + volume + downtrend
            elif close[i] < lowest_low[i] and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below lower Donchian
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above upper Donchian
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals