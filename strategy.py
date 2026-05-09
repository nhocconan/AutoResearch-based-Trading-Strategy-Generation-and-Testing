#!/usr/bin/env python3
name = "12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for each daily bar (R3, S3)
    # R3 = close + 1.1*(high-low)/6, S3 = close - 1.1*(high-low)/6
    camarilla_range = high_1d - low_1d
    R3 = close_1d + 1.1 * camarilla_range / 6
    S3 = close_1d - 1.1 * camarilla_range / 6
    
    # Align daily R3/S3 to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1-day average volume for volume spike detection
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for calculations
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-day average volume (aligned)
        volume_confirm = volume[i] > avg_volume_1d_aligned[i] * 1.5
        
        # Determine trend
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Enter long: price crosses above R3 + uptrend + volume confirmation
            if close[i] > R3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below S3 + downtrend + volume confirmation
            elif close[i] < S3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below S3 (reversion to mean)
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above R3 (reversion to mean)
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals