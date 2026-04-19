#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Breakout_1dVolume_1wTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian Channel (20) on 12h data
    # Using rolling window on high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max()
    donchian_lower = low_series.rolling(window=20, min_periods=20).min()
    donchian_upper_values = donchian_upper.values
    donchian_lower_values = donchian_lower.values
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_values[i]) or np.isnan(donchian_lower_values[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars, so divide by 2
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] / 2.0)
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + volume spike + above weekly EMA50
            if (close[i] > donchian_upper_values[i] and 
                volume_filter and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + volume spike + below weekly EMA50
            elif (close[i] < donchian_lower_values[i] and 
                  volume_filter and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to Donchian lower or trend weakens
            if close[i] < donchian_lower_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to Donchian upper or trend weakens
            if close[i] > donchian_upper_values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals