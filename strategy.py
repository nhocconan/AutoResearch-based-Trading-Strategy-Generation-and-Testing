#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_Breakout_Volume_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Momentum: ROC(5)
    roc = np.zeros_like(close)
    roc[5:] = (close[5:] - close[:-5]) / close[:-5] * 100
    
    # 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 7:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous week
    # Weekly high/low/close from 7 days ago
    weekly_high = df_1d['high'].shift(7).rolling(window=7, min_periods=7).max().values
    weekly_low = df_1d['low'].shift(7).rolling(window=7, min_periods=7).min().values
    weekly_close = df_1d['close'].shift(7).rolling(window=7, min_periods=7).last().values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Weekly support/resistance levels (standard pivot)
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    # Momentum filter: ROC > 0 for long, ROC < 0 for short
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume and positive momentum
            if (close[i] > weekly_r1_aligned[i] and 
                volume_confirm[i] and 
                roc[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume and negative momentum
            elif (close[i] < weekly_s1_aligned[i] and 
                  volume_confirm[i] and 
                  roc[i] < 0):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below weekly S1 or momentum turns negative
            if (close[i] < weekly_s1_aligned[i]) or (roc[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above weekly R1 or momentum turns positive
            if (close[i] > weekly_r1_aligned[i]) or (roc[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals