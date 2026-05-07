#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Use 1d Camarilla pivot R3/S3 levels for breakout signals on 6h timeframe. 
Breakout occurs when price closes outside R3/S3 with volume > 1.5x 20-period average. 
Trend filter uses 1w EMA(34) to align with higher timeframe momentum. 
This strategy targets 12-37 trades per year (50-150 total over 4 years) by requiring 
multiple confluences: price level breakout, volume surge, and trend alignment.
Designed to work in both bull and bear markets by using breakout logic (not pure trend following).
"""
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "6h"
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    # Using previous day's OHLC to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot point and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels: R3, S3, R4, S4
    r3 = pivot + (range_hl * 1.1000 / 2)  # R3 = pivot + (range * 1.1/2)
    s3 = pivot - (range_hl * 1.1000 / 2)  # S3 = pivot - (range * 1.1/2)
    r4 = pivot + (range_hl * 1.1000)      # R4 = pivot + (range * 1.1)
    s4 = pivot - (range_hl * 1.1000)      # S4 = pivot - (range * 1.1)
    
    # Al Camarilla levels to 6h timeframe (using previous day's values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(20, 34)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long breakout: price closes above R3 with volume and trend alignment
            if (close[i] > r3_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short breakdown: price closes below S3 with volume and trend alignment
            elif (close[i] < s3_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                if close[i] < s3_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r3_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_exit = 0
                else:
                    signals[i] = -0.25
    
    return signals