#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average). 
R1/S1 levels provide tighter breakouts than R3/S3 for better risk/reward. 
In bullish 1d trend: buy R1 breakouts, sell S1 breakdowns (continuation). 
In bearish 1d trend: sell S1 breakdowns, buy R1 breakouts (continuation). 
Volume filter ensures breakouts have conviction. 
Target: 100-180 total trades over 4 years = 25-45/year (within 75-200 target range).
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
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Camarilla pivot levels (dynamic per 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels using previous 4h bar's OHLC
    prev_close_4h = np.roll(df_4h['close'].values, 1)
    prev_high_4h = np.roll(df_4h['high'].values, 1)
    prev_low_4h = np.roll(df_4h['low'].values, 1)
    prev_close_4h[0] = df_4h['close'].values[0]
    prev_high_4h[0] = df_4h['high'].values[0]
    prev_low_4h[0] = df_4h['low'].values[0]
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla levels for 4h - using R1/S1 for tighter breakouts
    r1_4h = pivot_4h + (range_4h * 1.1 / 12)   # R1 level
    s1_4h = pivot_4h - (range_4h * 1.1 / 12)   # S1 level
    
    # Align 4h Camarilla levels to 4h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(34), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(pivot_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA34
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Breakout logic: trade continuation in direction of 1d trend with volume confirmation
            # In bullish 1d trend: buy R1 breakouts, sell S1 breakdowns
            # In bearish 1d trend: sell S1 breakdowns, buy R1 breakouts
            long_setup = (close[i] > r1_4h_aligned[i]) and htf_1d_bullish and volume_filter[i]
            short_setup = (close[i] < s1_4h_aligned[i]) and htf_1d_bearish and volume_filter[i]
            
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
            # Exit on trend reversal or mean reversion to 4h pivot
            exit_signal = (not htf_1d_bullish) or (close[i] < pivot_4h_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or mean reversion to 4h pivot
            exit_signal = htf_1d_bullish or (close[i] > pivot_4h_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0