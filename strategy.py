#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Volume spike: current 4h volume > 1.5 * 20-period average of 4h volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # Donchian channels (20-period) on 4h
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: close > weekly 20-period EMA for long, < for short
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    weekly_uptrend = close > ema_20_1w_aligned
    weekly_downtrend = close < ema_20_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume spike + weekly uptrend
            if close[i] > high_max[i] and vol_confirm and weekly_uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume spike + weekly downtrend
            elif close[i] < low_min[i] and vol_confirm and weekly_downtrend[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit when price breaks below Donchian lower
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit when price breaks above Donchian upper
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals