#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_With_12H_EMA34_Filter
Hypothesis: On 6h timeframe, break above/below daily Camarilla R1/S1 with volume confirmation, filtered by 12h EMA34 trend. Uses discrete position sizing (0.25) to limit risk. Targets 15-25 trades/year by requiring pivot break + volume + EMA filter, avoiding overtrading. Works in bull/bear via EMA trend filter and pivot structure as dynamic support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + ((h - l) * 1.1 / 2)
    r3 = c + ((h - l) * 1.1 / 4)
    r2 = c + ((h - l) * 1.1 / 6)
    r1 = c + ((h - l) * 1.1 / 12)
    s1 = c - ((h - l) * 1.1 / 12)
    s2 = c - ((h - l) * 1.1 / 6)
    s3 = c - ((h - l) * 1.1 / 4)
    s4 = c - ((h - l) * 1.1 / 2)
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1D bar
    r4_1d = np.full_like(close_1d, np.nan)
    r3_1d = np.full_like(close_1d, np.nan)
    r2_1d = np.full_like(close_1d, np.nan)
    r1_1d = np.full_like(close_1d, np.nan)
    s1_1d = np.full_like(close_1d, np.nan)
    s2_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r4_1d[i] = r4
        r3_1d[i] = r3
        r2_1d[i] = r2
        r1_1d[i] = r1
        s1_1d[i] = s1
        s2_1d[i] = s2
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get 12h data for EMA34 filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h
    if len(close_12h) >= 34:
        ema_12h = np.full_like(close_12h, np.nan)
        multiplier = 2 / (34 + 1)
        ema_12h[33] = np.mean(close_12h[0:34])
        for i in range(34, len(close_12h)):
            ema_12h[i] = (close_12h[i] - ema_12h[i-1]) * multiplier + ema_12h[i-1]
    else:
        ema_12h = np.full_like(close_12h, np.nan)
    
    # Align EMA34 to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate average volume for volume filter
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA34 and volume average
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(r2_1d_aligned[i]) or np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(avg_volume[i]) or
            volume[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: close > R1 and volume > average and price > EMA34(12h)
            if (close[i] > r1_1d_aligned[i] and 
                volume[i] > avg_volume[i] and 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: close < S1 and volume > average and price < EMA34(12h)
            elif (close[i] < s1_1d_aligned[i] and 
                  volume[i] > avg_volume[i] and 
                  close[i] < ema_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close < S1 or price < EMA34(12h)
            if (close[i] < s1_1d_aligned[i] or 
                close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close > R1 or price > EMA34(12h)
            if (close[i] > r1_1d_aligned[i] or 
                close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_With_12H_EMA34_Filter"
timeframe = "6h"
leverage = 1.0