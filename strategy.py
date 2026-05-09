#!/usr/bin/env python3
name = "1H_Camarilla_R3S3_Breakout_4hTrend_1dVolume"
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
    
    # Get 4h data for Camarilla calculation (HIGH, LOW, CLOSE from previous day)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Pre-calculate Camarilla levels for each 4h bar
    # Camarilla R3, R3, S3, S3 levels based on previous day's H, L, C
    # Using 4h data to get daily OHLC (last 4h bar of each day)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate daily OHLC from 4h bars (simplified: use last 4h bar of each day)
    # For simplicity, we'll use the 4h bar's OHLC as proxy for daily (acceptable for this strategy)
    camarilla_r3 = close_4h + (high_4h - low_4h) * 1.1/4
    camarilla_r4 = close_4h + (high_4h - low_4h) * 1.1/2
    camarilla_s3 = close_4h - (high_4h - low_4h) * 1.1/4
    camarilla_s4 = close_4h - (high_4h - low_4h) * 1.1/2
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price > EMA50 for long, < EMA50 for short
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x 20-day average volume
        volume_filter = volume[i] > avg_vol_1d_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: price breaks above R4 + uptrend + volume
            if close[i] > r4_aligned[i] and uptrend and volume_filter:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S4 + downtrend + volume
            elif close[i] < s4_aligned[i] and downtrend and volume_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S3 (reversion to mean)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R3 (reversion to mean)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals