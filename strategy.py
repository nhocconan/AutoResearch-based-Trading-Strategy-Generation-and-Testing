#!/usr/bin/env python3
"""
1h_4H1D_Camarilla_R1S1_Breakout_Trend_Filter
Hypothesis: 1h price breaks above/below 1D Camarilla R1/S1 levels with 4h EMA50 trend confirmation and volume spike.
Uses 1d for structure (Camarilla levels), 4h for trend direction (EMA50), 1h for entry timing and volume confirmation.
Targets 15-37 trades/year to minimize fee drag on 1h timeframe. Works in bull/bear markets via trend filter.
"""
name = "1h_4H1D_Camarilla_R1S1_Breakout_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1D Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 6)  # R1 level
    s1 = pivot - (range_1d * 1.1 / 6)  # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 4h data for trend direction (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_50 = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume filter: current 1h volume > 1.5 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (~8 hours on 1h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: price breaks above R1 with 4h EMA50 uptrend and volume spike
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                close[i] > ema_50_aligned[i] and volume_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_exit = 0
            # Short: price breaks below S1 with 4h EMA50 downtrend and volume spike
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  close[i] < ema_50_aligned[i] and volume_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA50 side (trend reversal)
            if position == 1 and close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals