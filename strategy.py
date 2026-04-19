#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for weekly pivot (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data for daily pivot (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points using last 5 days (Monday-Friday)
    high_5w = pd.Series(high_1w).rolling(window=5, min_periods=5).max().values
    low_5w = pd.Series(low_1w).rolling(window=5, min_periods=5).min().values
    close_5w = pd.Series(close_1w).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot calculation
    pivot_w = (high_5w + low_5w + close_5w) / 3.0
    r1_w = 2 * pivot_w - low_5w
    s1_w = 2 * pivot_w - high_5w
    
    # Calculate daily pivot points using previous day
    pivot_d = (high_1d + low_1d + close_1d) / 3.0
    r1_d = 2 * pivot_d - low_1d
    s1_d = 2 * pivot_d - high_1d
    
    # Align weekly pivot levels to 12h timeframe
    pivot_w_12h = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_12h = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_12h = align_htf_to_ltf(prices, df_1w, s1_w)
    
    # Align daily pivot levels to 12h timeframe
    pivot_d_12h = align_htf_to_ltf(prices, df_1d, pivot_d)
    r1_d_12h = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_d_12h = align_htf_to_ltf(prices, df_1d, s1_d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if np.isnan(pivot_w_12h[i]) or np.isnan(r1_w_12h[i]) or np.isnan(s1_w_12h[i]) or \
           np.isnan(pivot_d_12h[i]) or np.isnan(r1_d_12h[i]) or np.isnan(s1_d_12h[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above BOTH weekly R1 and daily R1 with volume
            if price > r1_w_12h[i] and price > r1_d_12h[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below BOTH weekly S1 and daily S1 with volume
            elif price < s1_w_12h[i] and price < s1_d_12h[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below weekly pivot OR daily pivot
            if price < pivot_w_12h[i] or price < pivot_d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above weekly pivot OR daily pivot
            if price > pivot_w_12h[i] or price > pivot_d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals