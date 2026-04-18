#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA34_Volume_Momentum
Hypothesis: Combines Camarilla R1/S1 breakouts with 12h EMA34 trend filter and volume momentum to capture strong directional moves. Uses tight entry conditions (breakout + volume + trend) to limit trades and reduce fee drag. Designed to work in both bull and bear markets by following higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        k = 2 / (34 + 1)
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = close_12h[i] * k + ema_34_12h[i-1] * (1 - k)
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from 1d OHLC
    def calculate_camarilla(high_val, low_val, close_val):
        range_val = high_val - low_val
        if range_val <= 0:
            return close_val, close_val, close_val, close_val
        multiplier = range_val * 1.1 / 12
        r1 = close_val + multiplier * 1.0
        s1 = close_val - multiplier * 1.0
        r2 = close_val + multiplier * 2.0
        s2 = close_val - multiplier * 2.0
        return r1, s1, r2, s2
    
    # Vectorized Camarilla calculation for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r1_1d = np.full(len(close_1d), np.nan)
    s1_1d = np.full(len(close_1d), np.nan)
    r2_1d = np.full(len(close_1d), np.nan)
    s2_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        r1, s1, r2, s2 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        s1_1d[i] = s1
        r2_1d[i] = r2
        s2_1d[i] = s2
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    # Volume momentum: current volume > 1.8x 24-period average
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    vol_momentum = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 40  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price breaks above R1 with volume momentum and above 12h EMA34
            if close[i] > r1_1d_aligned[i] and vol_momentum[i] and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price breaks below S1 with volume momentum and below 12h EMA34
            elif close[i] < s1_1d_aligned[i] and vol_momentum[i] and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 4 bars hold, then exit on trend reversal or volume drop
            if bars_since_entry >= 4:
                if close[i] < ema_34_12h_aligned[i] or not vol_momentum[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 4 bars hold, then exit on trend reversal or volume drop
            if bars_since_entry >= 4:
                if close[i] > ema_34_12h_aligned[i] or not vol_momentum[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Volume_Momentum"
timeframe = "4h"
leverage = 1.0