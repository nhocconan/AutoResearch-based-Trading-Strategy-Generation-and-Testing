#!/usr/bin/env python3
# 12h_daily_camarilla_pullback_v2
# Hypothesis: 12h strategy using daily Camarilla pivot levels with mean reversion.
# Long: Price pulls back to daily S3 level with volume < 0.8x 20-period average (low volume pullback).
# Short: Price pulls back to daily R3 level with volume < 0.8x 20-period average.
# Exit: Price returns to daily pivot point (PP).
# Uses daily Camarilla for key support/resistance, 12h for execution, low volume for pullback confirmation.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_daily_camarilla_pullback_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = close_1d + range_1d * 1.1 / 4.0
    s3 = close_1d - range_1d * 1.1 / 4.0
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Low volume confirmation: current volume < 0.8x 20-period average (pullback on weak volume)
        low_volume = volume[i] < 0.8 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to daily pivot
            if close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to daily pivot
            if close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for pullback to S3/R3 with low volume confirmation
            pullback_to_support = (abs(close[i] - s3_aligned[i]) < 0.005 * close[i]) and low_volume
            pullback_to_resistance = (abs(close[i] - r3_aligned[i]) < 0.005 * close[i]) and low_volume
            
            if pullback_to_support:
                position = 1
                signals[i] = 0.25
            elif pullback_to_resistance:
                position = -1
                signals[i] = -0.25
    
    return signals