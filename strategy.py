#!/usr/bin/env python3
# 6h_1w_1d_donchian_breakout_v1
# Hypothesis: Breakout strategy using daily Donchian channels (20-period) and weekly trend filter (EMA50).
# Enter long when price breaks above daily upper Donchian band, price > weekly EMA50, and volume > 1.5x average volume.
# Enter short when price breaks below daily lower Donchian band, price < weekly EMA50, and volume > 1.5x average volume.
# Exit when price returns to the opposite Donchian band or weekly trend filter fails.
# Designed for 15-40 trades/year on 6h to avoid fee drag. Works in bull/bear via weekly trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels for each daily bar
    upper_dc = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian channels to 6h timeframe
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, lower_dc)
    
    # Volume average (20-period) for confirmation
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_dc_aligned[i]) or np.isnan(lower_dc_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirmed = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit: price returns to lower Donchian band or weekly trend fails
            if close[i] < lower_dc_aligned[i] or close[i] <= ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to upper Donchian band or weekly trend fails
            if close[i] > upper_dc_aligned[i] or close[i] >= ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above daily upper Donchian with volume and trend filter
            if (close[i] > upper_dc_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                vol_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below daily lower Donchian with volume and trend filter
            elif (close[i] < lower_dc_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  vol_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals