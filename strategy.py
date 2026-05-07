#!/usr/bin/env python3
"""
12H_PriceChannel_Turnaround_v1
Hypothesis: Use 1-day Donchian channel (20-period) for trend context and 12-hour price reversals at channel extremes for entries.
Long when 12h price closes below prior 12h low AND 1d Donchian lower band (20) is sloping up; 
Short when 12h price closes above prior 12h high AND 1d Donchian upper band (20) is sloping down.
Volume confirmation: current 12h volume > 1.3x 20-period average volume.
This captures mean-reversion turns within the larger trend, working in both ranging and trending markets.
"""
name = "12H_PriceChannel_Turnaround_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    dh_up = donchian_high  # upper band
    dh_low = donchian_low  # lower band
    dh_up_aligned = align_htf_to_ltf(prices, df_1d, dh_up)
    dh_low_aligned = align_htf_to_ltf(prices, df_1d, dh_low)
    
    # Slope of Donchian bands (trend filter)
    dh_up_slope = np.diff(dh_up_aligned, prepend=dh_up_aligned[0])
    dh_low_slope = np.diff(dh_low_aligned, prepend=dh_low_aligned[0])
    
    # Volume filter: current 12h volume > 1.3x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 20)  # Ensure sufficient warmup for Donchian
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(dh_up_aligned[i]) or np.isnan(dh_low_aligned[i]) or 
            np.isnan(dh_up_slope[i]) or np.isnan(dh_low_slope[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 6 bars between trades (3 days on 12h TF) to reduce frequency
            if bars_since_exit < 6:
                continue
                
            # Long: 12h price closes below prior 12h low AND 1d Donchian lower band sloping up
            if (close[i] < low[i-1] and 
                dh_low_slope[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: 12h price closes above prior 12h high AND 1d Donchian upper band sloping down
            elif (close[i] > high[i-1] and 
                  dh_up_slope[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to prior 12h level (mean reversion complete)
            if position == 1 and close[i] >= high[i-1]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] <= low[i-1]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals