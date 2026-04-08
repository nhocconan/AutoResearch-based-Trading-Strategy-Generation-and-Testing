#!/usr/bin/env python3
"""
4h_1d1w_camarilla_pivot_v1
Hypothesis: Trade reversals at daily Camarilla pivot levels with weekly trend filter and volume confirmation.
- Long when price touches S3 level in weekly uptrend with volume confirmation
- Short when price touches R3 level in weekly downtrend with volume confirmation
- Uses weekly EMA for trend filter and daily Camarilla levels for entries
- Designed for low trade frequency (20-40/year) to minimize fee drag
- Works in bull/bear via trend filter and mean-reversion at extreme levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d1w_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    range_val = high - low
    if range_val <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Camarilla levels
    S1 = close - (range_val * 1.0 / 12)
    S2 = close - (range_val * 2.0 / 12)
    S3 = close - (range_val * 3.0 / 12)
    S4 = close - (range_val * 4.0 / 12)
    R1 = close + (range_val * 1.0 / 12)
    R2 = close + (range_val * 2.0 / 12)
    R3 = close + (range_val * 3.0 / 12)
    R4 = close + (range_val * 4.0 / 12)
    
    return S1, S2, S3, S4, R1, R2, R3, R4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    S1 = np.full(len(close_1d), np.nan)
    S2 = np.full(len(close_1d), np.nan)
    S3 = np.full(len(close_1d), np.nan)
    S4 = np.full(len(close_1d), np.nan)
    R1 = np.full(len(close_1d), np.nan)
    R2 = np.full(len(close_1d), np.nan)
    R3 = np.full(len(close_1d), np.nan)
    R4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        s1, s2, s3, s4, r1, r2, r3, r4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        S1[i] = s1
        S2[i] = s2
        S3[i] = s3
        S4[i] = s4
        R1[i] = r1
        R2[i] = r2
        R3[i] = r3
        R4[i] = r4
    
    # Calculate 1w EMA (20-period) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        alpha = 2.0 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    
    # Align indicators to 4h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: 20-period average on 4h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(S3_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        s3_level = S3_aligned[i]
        r3_level = R3_aligned[i]
        trend_up = price > ema_20_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price moves back above S2 or trend turns down
            s2_level = align_htf_to_ltf(prices, df_1d, 
                                       np.full(len(close_1d), np.nan))[i]
            # Calculate S2 for current day
            if i < len(S2):
                s2_level = S2_aligned[i]
            if price > s2_level or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price moves back below R2 or trend turns up
            r2_level = align_htf_to_ltf(prices, df_1d, 
                                       np.full(len(close_1d), np.nan))[i]
            # Calculate R2 for current day
            if i < len(R2):
                r2_level = R2_aligned[i]
            if price < r2_level or trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches S3 in uptrend with volume confirmation
            if price <= s3_level and trend_up and vol_ratio > 1.3:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches R3 in downtrend with volume confirmation
            elif price >= r3_level and not trend_up and vol_ratio > 1.3:
                position = -1
                signals[i] = -0.25
    
    return signals