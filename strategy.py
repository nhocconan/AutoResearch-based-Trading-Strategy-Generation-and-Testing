#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for weekly high/low
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly high and low (using previous week's values)
    weekly_high_prev = np.roll(high_1w, 1)
    weekly_low_prev = np.roll(low_1w, 1)
    weekly_high_prev[0] = np.nan  # First value is invalid
    weekly_low_prev[0] = np.nan
    
    # Align weekly levels to 6h timeframe
    weekly_high_6h = align_htf_to_ltf(prices, df_1w, weekly_high_prev)
    weekly_low_6h = align_htf_to_ltf(prices, df_1w, weekly_low_prev)
    
    # Daily trend: EMA50 on 1d
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: volume > 1.5x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_6h[i]) or np.isnan(weekly_low_6h[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: weekly trend breaks or price below weekly low
            if close[i] < weekly_low_6h[i] or ema_50_aligned[i] < ema_50_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly trend breaks or price above weekly high
            if close[i] > weekly_high_6h[i] or ema_50_aligned[i] > ema_50_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly high + uptrend + volume
            if (close[i] > weekly_high_6h[i] and 
                ema_50_aligned[i] > ema_50_aligned[i-1] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly low + downtrend + volume
            elif (close[i] < weekly_low_6h[i] and 
                  ema_50_aligned[i] < ema_50_aligned[i-1] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals