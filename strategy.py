#!/usr/bin/env python3
"""
12h_Pivot_R2S2_Breakout_VolumeFilter_V1
Hypothesis: Price breaking above R2 or below S2 pivot levels on 12h with volume confirmation provides high-probability trend continuation. Uses 1d pivots calculated from prior day's OHLC. Works in bull/bear by taking breakouts in either direction with volume filter to avoid false breaks in ranging markets. Targets 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot_points(high, low, close):
    """Calculate pivot points and support/resistance levels"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, s1, r2, s2

def generate_signals(prices):
    n = len(prices)
    if n < 2:
        return np.zeros(n)
    
    # Load 1d data once for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's data to calculate today's pivot (no look-ahead)
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        pivot_1d[i], r1_1d[i], s1_1d[i], r2_1d[i], s2_1d[i] = calculate_pivot_points(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
    
    # Align pivot levels to 12h timeframe (using prior day's values)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume confirmation - volume above 20-period average
    volume = prices['volume'].values
    vol_ma = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if NaN in critical values
        if np.isnan(pivot_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].values[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above R2 with volume
            if price > r2_12h[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume
            elif price < s2_12h[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot level
            if price <= pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot level
            if price >= pivot_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R2S2_Breakout_VolumeFilter_V1"
timeframe = "12h"
leverage = 1.0