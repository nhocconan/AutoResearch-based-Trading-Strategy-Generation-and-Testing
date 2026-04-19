#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Conditioned_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(34) for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Donchian channels (20)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume confirmation (volume > 1.5x 20-day average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + weekly uptrend + volume
            if (close[i] > high_20[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + weekly downtrend + volume
            elif (close[i] < low_20[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price breaks below Donchian low OR trend changes
            if (close[i] < low_20[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price breaks above Donchian high OR trend changes
            if (close[i] > high_20[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals