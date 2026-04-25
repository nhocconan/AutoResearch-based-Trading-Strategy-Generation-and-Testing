#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 1d EMA50 trend filter and volume confirmation. R3/S3 levels represent stronger support/resistance, reducing false breakouts. Works in both bull and bear markets by aligning with 1d trend - in bullish 1d trend we buy R3 breakouts and sell S3 breakdowns (continuation), in bearish 1d trend we sell S3 breakdowns and buy R3 breakouts (continuation). Volume spike requirement (>2.0x 20-period average) filters low-quality breakouts. Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 6h data for Camarilla pivot levels (dynamic per 6h bar)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate 6h Camarilla pivot levels using previous 6h bar's OHLC
    prev_close_6h = np.roll(df_6h['close'].values, 1)
    prev_high_6h = np.roll(df_6h['high'].values, 1)
    prev_low_6h = np.roll(df_6h['low'].values, 1)
    prev_close_6h[0] = df_6h['close'].values[0]
    prev_high_6h[0] = df_6h['high'].values[0]
    prev_low_6h[0] = df_6h['low'].values[0]
    
    pivot_6h = (prev_high_6h + prev_low_6h + prev_close_6h) / 3.0
    range_6h = prev_high_6h - prev_low_6h
    
    # Camarilla levels for 6h - using R3/S3 for stronger breakouts
    r3_6h = pivot_6h + (range_6h * 1.1 / 4)   # R3 level
    s3_6h = pivot_6h - (range_6h * 1.1 / 4)   # S3 level
    
    # Align 6h Camarilla levels to 6h timeframe
    r3_6h_aligned = align_htf_to_ltf(prices, df_6h, r3_6h)
    s3_6h_aligned = align_htf_to_ltf(prices, df_6h, s3_6h)
    pivot_6h_aligned = align_htf_to_ltf(prices, df_6h, pivot_6h)
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_6h_aligned[i]) or
            np.isnan(s3_6h_aligned[i]) or
            np.isnan(pivot_6h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA50
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Breakout logic: trade continuation in direction of 1d trend with volume confirmation
            # In bullish 1d trend: buy R3 breakouts, sell S3 breakdowns
            # In bearish 1d trend: sell S3 breakdowns, buy R3 breakouts
            long_setup = (close[i] > r3_6h_aligned[i]) and htf_1d_bullish and volume_spike[i]
            short_setup = (close[i] < s3_6h_aligned[i]) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or mean reversion to 6h pivot
            exit_signal = (not htf_1d_bullish) or (close[i] < pivot_6h_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to 6h pivot
            exit_signal = htf_1d_bullish or (close[i] > pivot_6h_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0