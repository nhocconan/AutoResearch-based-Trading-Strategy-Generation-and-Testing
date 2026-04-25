#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume spike confirmation.
R4/S4 represent stronger breakout levels than R3/S3, reducing false signals and improving trade quality.
Long when price breaks above R4 with 1d EMA50 uptrend and volume > 2.0x 20-period average.
Short when price breaks below S4 with 1d EMA50 downtrend and volume > 2.0x 20-period average.
Exit on opposite band touch (S4/R4) or trend reversal.
Uses discrete sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
Works in bull via trend-following breakouts, in bear via reduced whipsaw from stronger breakout levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 5:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Camarilla levels for each 6h bar (based on previous bar)
    R4_6h = np.full(len(close_6h), np.nan)
    S4_6h = np.full(len(close_6h), np.nan)
    R3_6h = np.full(len(close_6h), np.nan)
    S3_6h = np.full(len(close_6h), np.nan)
    
    for i in range(1, len(close_6h)):
        # Camarilla levels based on previous 6h bar's range
        high_prev = high_6h[i-1]
        low_prev = low_6h[i-1]
        close_prev = close_6h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R4_6h[i] = close_prev + (range_prev * 1.1 / 2)  # R4 level
            S4_6h[i] = close_prev - (range_prev * 1.1 / 2)  # S4 level
            R3_6h[i] = close_prev + (range_prev * 1.1 / 4)  # R3 level
            S3_6h[i] = close_prev - (range_prev * 1.1 / 4)  # S3 level
    
    # Align Camarilla levels to original timeframe
    R4_6h_aligned = align_htf_to_ltf(prices, df_6h, R4_6h)
    S4_6h_aligned = align_htf_to_ltf(prices, df_6h, S4_6h)
    R3_6h_aligned = align_htf_to_ltf(prices, df_6h, R3_6h)
    S3_6h_aligned = align_htf_to_ltf(prices, df_6h, S3_6h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R4_6h_aligned[i]) or np.isnan(S4_6h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R4 with uptrend and volume spike
            long_signal = (close[i] > R4_6h_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below S4 with downtrend and volume spike
            short_signal = (close[i] < S4_6h_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price touches S3 or trend reverses
            exit_signal = (close[i] < S3_6h_aligned[i]) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches R3 or trend reverses
            exit_signal = (close[i] > R3_6h_aligned[i]) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R4S4_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0