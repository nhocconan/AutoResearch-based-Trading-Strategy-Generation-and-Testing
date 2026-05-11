#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day (H, L, C)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Daily trend: EMA34
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = df_1d['close'] > ema34
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + daily uptrend + volume spike
            if (close[i] > R1_aligned[i] and 
                trend_up_aligned[i] and 
                volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + daily downtrend + volume spike
            elif (close[i] < S1_aligned[i] and 
                  not trend_up_aligned[i] and 
                  volume[i] > 2.0 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend changes
            if (close[i] < S1_aligned[i] or not trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend changes
            if (close[i] > R1_aligned[i] or trend_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals