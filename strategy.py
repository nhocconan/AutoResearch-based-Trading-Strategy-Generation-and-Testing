#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R1S1_Breakout_Volume_EMA34Filter_v1
Hypothesis: Camarilla pivot breakout with volume confirmation and EMA34 filter on 1d timeframe.
Go long when price breaks above R1 with volume > 1.5x average and price > 1d EMA34.
Go short when price breaks below S1 with volume > 1.5x average and price < 1d EMA34.
Uses 6h timeframe to reduce trade frequency and avoid excessive fees.
Designed to work in both bull and bear markets by capturing breakouts with trend filter.
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
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Using (H+L+C)/3 as pivot, then R1/S1 calculations
    pivot_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    
    if len(high_1d) >= 1:
        for i in range(1, len(high_1d)):
            # Use previous day's data for today's levels
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            
            pivot_1d[i] = pivot
            r1_1d[i] = pivot + (range_val * 1.0 / 12.0)  # R1
            s1_1d[i] = pivot - (range_val * 1.0 / 12.0)  # S1
    
    # Calculate 1d EMA34
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).values
    
    # Align 1d indicators to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(1, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and above EMA34
            if close[i] > r1_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and below EMA34
            elif close[i] < s1_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or below EMA34
            if close[i] < s1_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or above EMA34
            if close[i] > r1_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_R1S1_Breakout_Volume_EMA34Filter_v1"
timeframe = "6h"
leverage = 1.0