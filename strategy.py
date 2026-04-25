#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm_v2
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and 1d EMA50 trend confirmation.
Long when price breaks above 6h Donchian upper band with price above weekly pivot and 1d EMA50 uptrend.
Short when price breaks below 6h Donchian lower band with price below weekly pivot and 1d EMA50 downtrend.
Volume confirmation required. Uses discrete sizing (0.25) to minimize fees.
Target: 50-150 total trades over 4 years = 12-37/year. Works in bull via trend-following breakouts,
in bear via mean reversion at bands with strong trend filter to avoid whipsaw.
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
    
    # Get 6h data for Donchian calculations (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate Donchian channels for each 6h bar (based on previous 20 bars)
    upper_6h = np.full(len(close_6h), np.nan)
    lower_6h = np.full(len(close_6h), np.nan)
    
    for i in range(20, len(close_6h)):
        # Donchian channels based on previous 20 6h bars
        high_prev = high_6h[i-20:i]
        low_prev = low_6h[i-20:i]
        
        upper_6h[i] = np.max(high_prev)
        lower_6h[i] = np.min(low_prev)
    
    # Align Donchian levels to original timeframe
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    
    # Get 1w data for weekly pivot point (based on previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point (standard: (H+L+C)/3)
    pivot_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        pivot_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
    
    # Align weekly pivot to original timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_6h_aligned[i]) or np.isnan(lower_6h_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above upper band with price above pivot and uptrend and volume spike
            long_signal = (close[i] > upper_6h_aligned[i]) and (close[i] > pivot_1w_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and vol_spike[i]
            # Short: price breaks below lower band with price below pivot and downtrend and volume spike
            short_signal = (close[i] < lower_6h_aligned[i]) and (close[i] < pivot_1w_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and vol_spike[i]
            
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
            # Exit conditions: price touches lower band or trend reverses
            exit_signal = (close[i] < lower_6h_aligned[i]) or (close[i] < ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches upper band or trend reverses
            exit_signal = (close[i] > upper_6h_aligned[i]) or (close[i] > ema_50_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0