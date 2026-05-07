#!/usr/bin/env python3
# 1H_Camarilla_R3S3_4hTrend_VolumeSpike
# Hypothesis: Uses 4h trend direction (above/below 4h EMA50) as signal filter, with 1h entries on 4h-derived Camarilla R3/S3 breakouts confirmed by volume spikes. Targets 15-30 trades/year by requiring 4h trend alignment, 1h price break of Camarilla levels, and volume > 2x 20-period average. Designed to work in both bull and bear markets by following the 4h trend.

name = "1H_Camarilla_R3S3_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 4h Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    r3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4
    s3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume spike: current volume > 2x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R3 + uptrend (close > 4h EMA50) + volume spike
            if (close[i] > r3_4h_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and
                volume_filter):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 + downtrend (close < 4h EMA50) + volume spike
            elif (close[i] < s3_4h_aligned[i] and 
                  close[i] < ema50_4h_aligned[i] and
                  volume_filter):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns below 4h EMA50 or to S3 level
            if close[i] < ema50_4h_aligned[i] or close[i] < s3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns above 4h EMA50 or to R3 level
            if close[i] > ema50_4h_aligned[i] or close[i] > r3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals