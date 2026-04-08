#!/usr/bin/env python3
"""
6h_12h1d_camarilla_pivot_v1
Hypothesis: Use 12h/1d Camarilla pivot levels on 6h chart with breakout/fade logic.
- In trending markets: break above R4 or below S4 signals continuation (trend follow)
- In ranging markets: fade at R3/S3 levels (mean reversion)
- Uses 12h trend filter to determine regime
- Volume confirms breakouts
Target: 15-25 trades/year per symbol to avoid overtrading.
Works in both bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h1d_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    # Camarilla levels
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
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Arrays to hold Camarilla levels for each 12h bar
    r4_12h = np.full(len(close_12h), np.nan)
    r3_12h = np.full(len(close_12h), np.nan)
    s3_12h = np.full(len(close_12h), np.nan)
    s4_12h = np.full(len(close_12h), np.nan)
    
    for i in range(len(close_12h)):
        r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla(high_12h[i], low_12h[i], close_12h[i])
        r4_12h[i] = r4
        r3_12h[i] = r3
        s3_12h[i] = s3
        s4_12h[i] = s4
    
    # Align Camarilla levels to 6h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # 12h trend filter: 20-period EMA
    ema_20_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        # Calculate EMA manually
        alpha = 2.0 / (20 + 1)
        ema_20_12h[19] = np.mean(close_12h[:20])  # SMA for first value
        for i in range(20, len(close_12h)):
            ema_20_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_20_12h[i-1]
    
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume confirmation: 20-period average on 6h
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        r4 = r4_12h_aligned[i]
        r3 = r3_12h_aligned[i]
        s3 = s3_12h_aligned[i]
        s4 = s4_12h_aligned[i]
        trend_up = price > ema_20_12h_aligned[i]
        
        if position == 1:  # Long
            # Exit: price breaks below S3 or trend turns down
            if price < s3 or not trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks above R3 or trend turns up
            if price > r3 or trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Determine regime: trending if price beyond R4/S4
            if price > r4 and vol_ratio > 1.8:  # Strong breakout up
                position = 1
                signals[i] = 0.25
            elif price < s4 and vol_ratio > 1.8:  # Strong breakdown down
                position = -1
                signals[i] = -0.25
            # Ranging market: fade at R3/S3
            elif price > r3 and vol_ratio > 1.2:  # Sell at R3
                position = -1
                signals[i] = -0.25
            elif price < s3 and vol_ratio > 1.2:  # Buy at S3
                position = 1
                signals[i] = 0.25
    
    return signals