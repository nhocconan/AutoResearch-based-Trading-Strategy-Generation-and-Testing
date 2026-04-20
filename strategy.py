#!/usr/bin/env python3
"""
4h_1d_Pivot_R1S1_Breakout_Volume
Hypothesis: Use daily Camarilla pivot levels (R1/S1) from 1d timeframe as support/resistance.
Breakout above R1 with volume confirmation = long; breakdown below S1 with volume confirmation = short.
Exit at opposite pivot level (S1 for longs, R1 for shorts). Only trade during London/New York session (8-20 UTC).
Daily pivot provides structural levels that work in both bull and bear markets; volume confirms breakout validity.
Target: 20-50 trades per year with position size 0.25 to manage drawdown.
"""

name = "4h_1d_Pivot_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close
    close_val = close
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    return r1, s1, close_val, close_val  # R1, S1, R3, S3 (we only need R1/S1)

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla R1 and S1
    r1_1d = np.full(len(close_1d), np.nan)
    s1_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        r1, s1, _, _ = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Align to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_ratio = volume / vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above R1 with volume confirmation
            if close[i] > r1_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation
            elif close[i] < s1_1d_aligned[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches S1 (opposite level)
            if close[i] <= s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches R1 (opposite level)
            if close[i] >= r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals