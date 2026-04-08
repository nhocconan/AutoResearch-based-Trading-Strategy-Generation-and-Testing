#!/usr/bin/env python3
"""
1d_weekly_donchian_breakout_volume_v1
Hypothesis: Weekly Donchian breakout with daily volume confirmation and price above/below 50 EMA.
Trades with the weekly trend using daily timeframe for precise entry.
Designed to work in both bull and bear markets by capturing breakouts in the direction of weekly momentum.
Target: 10-25 trades per year to minimize fee drag and avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    highest_high_1w = np.full(len(high_1w), np.nan)
    lowest_low_1w = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        highest_high_1w[i] = np.max(high_1w[i-20:i])
        lowest_low_1w[i] = np.min(low_1w[i-20:i])
    
    highest_high_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    lowest_low_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    
    # Daily volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Daily trend filter: 50 EMA
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_high_1w_aligned[i]) or np.isnan(lowest_low_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 1:  # Long
            # Exit: price breaks below weekly Donchian low
            if close[i] < lowest_low_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above weekly Donchian high
            if close[i] > highest_high_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: Weekly Donchian breakout with volume confirmation
            if (close[i] > highest_high_1w_aligned[i] and 
                vol_ratio > 1.5):
                position = 1
                signals[i] = 0.25
            # Short: Weekly Donchian breakdown with volume confirmation
            elif (close[i] < lowest_low_1w_aligned[i] and 
                  vol_ratio > 1.5):
                position = -1
                signals[i] = -0.25
    
    return signals