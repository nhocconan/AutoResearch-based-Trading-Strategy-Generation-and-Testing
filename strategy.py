#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_With_Volume
Breakout strategy using weekly pivot levels on 6h timeframe.
Long when price breaks above weekly R1 with volume confirmation and price > 1d EMA50.
Short when price breaks below weekly S1 with volume confirmation and price < 1d EMA50.
Exit when price returns to weekly pivot (PP) or trend filter fails.
Uses weekly pivot for structure and 1d EMA for trend filter to avoid counter-trend trades.
Target: 15-30 trades/year per symbol.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = (2 * Pivot) - Low
    r1_1w = (2 * pivot_1w) - low_1w
    # S1 = (2 * Pivot) - High
    s1_1w = (2 * pivot_1w) - high_1w
    
    # Align weekly pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    if n >= vol_period:
        for i in range(vol_period - 1, n):
            vol_ma[i] = np.mean(volume[i - vol_period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot, EMA1d, and volume MA
    start_idx = max(0, ema_1d_period - 1, vol_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_avg = vol_ma[i]
        ema1d_val = ema_1d_aligned[i]
        pivot_val = pivot_1w_aligned[i]
        r1_val = r1_1w_aligned[i]
        s1_val = s1_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume > 1.5x average and price > 1d EMA50
            if (price > r1_val and vol > 1.5 * vol_avg and price > ema1d_val):
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume > 1.5x average and price < 1d EMA50
            elif (price < s1_val and vol > 1.5 * vol_avg and price < ema1d_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to pivot or trend fails
            if (price <= pivot_val) or price < ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to pivot or trend fails
            if (price >= pivot_val) or price > ema1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_Breakout_With_Volume"
timeframe = "6h"
leverage = 1.0