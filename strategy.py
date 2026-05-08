#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3
    range_1d = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels
    r3 = pivot_1d + range_1d * 1.1 / 4
    s3 = pivot_1d - range_1d * 1.1 / 4
    r4 = pivot_1d + range_1d * 1.1 / 2
    s4 = pivot_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above weekly EMA
            if (close[i] > r3_aligned[i] and vol_spike and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike, below weekly EMA
            elif (close[i] < s3_aligned[i] and vol_spike and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or below weekly EMA
            if (close[i] < s3_aligned[i] or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or above weekly EMA
            if (close[i] > r3_aligned[i] or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines Camarilla R3/S3 breakouts with volume spike confirmation and weekly trend filter.
# - Uses daily Camarilla levels (R3/S3) for entry/exit levels
# - Requires volume spike (>2x 20-period average) for confirmation
# - Uses weekly EMA(50) as trend filter (only long above EMA, short below EMA)
# - Designed for 12h timeframe to capture medium-term breaks with confirmation
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return
# - Works in both bull and bear markets by following weekly trend direction
# - Volume spike reduces false breakouts from low-volume moves
# - Camarilla levels provide institutional-grade support/resistance levels