#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_VolumeFilter_V2
Hypothesis: Camarilla pivot R1/S1 breakouts with volume confirmation work well on 12h timeframe.
Only trade breakouts in direction of 1w trend (EMA50) to avoid counter-trend whipsaw.
Volume must be > 1.5x average volume for confirmation. Designed for fewer trades (<40/year)
to minimize fee drag while capturing significant moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = np.zeros_like(close_1d)
    r1_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        pivot_1d[i], r1_1d[i], s1_1d[i] = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = np.zeros_like(close_1w)
    ema_50_1w[:] = np.nan
    
    if len(close_1w) >= 50:
        # Calculate EMA manually with proper initialization
        alpha = 2.0 / (50 + 1)
        ema_50_1w[49] = np.mean(close_1w[:50])  # Simple average for first value
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # Align EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 1.5x average volume
    volume = prices['volume'].values
    vol_ma = np.zeros_like(volume)
    vol_ma[:] = np.nan
    
    if len(volume) >= 20:
        # Calculate 20-period moving average of volume
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        uptrend = price > ema_50_aligned[i]
        downtrend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + uptrend
            if price > r1_aligned[i] and vol_ratio > 1.5 and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + downtrend
            elif price < s1_aligned[i] and vol_ratio > 1.5 and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below pivot or volume drops
            if price < pivot_aligned[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above pivot or volume drops
            if price > pivot_aligned[i] or vol_ratio < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_VolumeFilter_V2"
timeframe = "12h"
leverage = 1.0