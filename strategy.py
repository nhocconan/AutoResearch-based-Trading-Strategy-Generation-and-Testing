#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Trade Camarilla R3/S3 breakouts on 1d with 1w EMA50 trend filter and volume confirmation (1.5x 5-bar avg). 
Designed for 1d timeframe targeting 7-25 trades/year. Uses strong breakout levels (R3/S3) with trend alignment 
to work in both bull and bear markets by following 1w trend. Volume confirmation reduces false breakouts.
"""

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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Get 1d data for Camarilla levels (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_range = high_1d - low_1d
    r3 = close_1d + 1.1 * camarilla_range
    s3 = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 1d timeframe (1-day lagged for previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 5-bar average volume
    volume_ma = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and volume MA
    start_idx = max(50, 5)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend: price above/below EMA50
        trend_bullish = close[i] > ema_50_aligned[i]
        trend_bearish = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Look for breakout signals with volume confirmation and trend alignment
            # Require close outside R3/S3 (not just touching) for stronger breakout confirmation
            long_signal = close[i] > r3_aligned[i] and volume_spike[i] and trend_bullish
            short_signal = close[i] < s3_aligned[i] and volume_spike[i] and trend_bearish
            
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
            # Exit when price breaks below S3 or trend reverses
            exit_signal = close[i] < s3_aligned[i] or not trend_bullish
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above R3 or trend reverses
            exit_signal = close[i] > r3_aligned[i] or not trend_bearish
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0