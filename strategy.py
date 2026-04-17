#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_1dTrendFilter_V1
Strategy: 12h Camarilla pivot breakout with volume confirmation and 1d trend filter.
Long: Break above R3 with volume > 1.5x MA(20) and price above 1d EMA(34)
Short: Break below S3 with volume > 1.5x MA(20) and price below 1d EMA(34)
Exit: Price returns to Pivot level or trend filter fails
Position size: 0.25
Designed to capture institutional breakouts while avoiding false signals in low volatility.
Timeframe: 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 12h bar
    # Formula: Pivot = (H+L+C)/3
    # R3 = Pivot + 1.1*(H-L), S3 = Pivot - 1.1*(H-L)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    pivot_12h = (h_12h + l_12h + c_12h) / 3.0
    range_12h = h_12h - l_12h
    r3_12h = pivot_12h + 1.1 * range_12h
    s3_12h = pivot_12h - 1.1 * range_12h
    
    # Align to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Volume filter
    volume_ma20 = np.convolve(volume, np.ones(20)/20, mode='full')[:len(volume)]
    volume_ma20 = np.concatenate([np.full(19, np.nan), volume_ma20[19:]])
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_ma20[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Entry signals
        if position == 0:
            # Long: Break above R3 with volume and trend confirmation
            if close[i] > r3_aligned[i] and volume_filter and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume and trend confirmation
            elif close[i] < s3_aligned[i] and volume_filter and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to pivot or trend filter fails
            if close[i] <= pivot_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to pivot or trend filter fails
            if close[i] >= pivot_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_1dTrendFilter_V1"
timeframe = "12h"
leverage = 1.0