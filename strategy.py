#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Volume_Squeeze
Strategy: 4h Camarilla pivot level breakouts with volume squeeze filter and 12h trend.
Long: Break above R1 with volume contraction followed by expansion + 12h uptrend
Short: Break below S1 with volume contraction followed by expansion + 12h downtrend
Exit: Price returns to Pivot point or trend reversal
Position size: 0.25
Designed to work in both trending and ranging markets by combining volatility contraction/expansion with institutional levels.
Timeframe: 4h
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
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align daily levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 12h trend (bullish/bearish)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    trend_12h = (df_12h['close'] > df_12h['open']).astype(float).values  # 1 for up, 0 for down
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume squeeze detection: look for low volume followed by expansion
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.maximum(volume_ma20, 1e-10)  # Avoid division by zero
    
    # Volume contraction: current volume < 70% of 20-period average
    # Volume expansion: current volume > 120% of 20-period average
    volume_contraction = volume_ratio < 0.7
    volume_expansion = volume_ratio > 1.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(20, n):  # warmup for volume MA
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(trend_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: Price breaks above R1 with volume expansion after contraction + 12h uptrend
            if (close[i-1] <= r1_aligned[i-1] and close[i] > r1_aligned[i] and
                volume_expansion[i] and 
                np.any(volume_contraction[max(0, i-5):i]) and  # Contraction in last 5 bars
                trend_12h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume expansion after contraction + 12h downtrend
            elif (close[i-1] >= s1_aligned[i-1] and close[i] < s1_aligned[i] and
                  volume_expansion[i] and 
                  np.any(volume_contraction[max(0, i-5):i]) and  # Contraction in last 5 bars
                  trend_12h_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to pivot or 12h trend turns down
            if close[i] <= pivot_aligned[i] or trend_12h_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to pivot or 12h trend turns up
            if close[i] >= pivot_aligned[i] or trend_12h_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Volume_Squeeze"
timeframe = "4h"
leverage = 1.0