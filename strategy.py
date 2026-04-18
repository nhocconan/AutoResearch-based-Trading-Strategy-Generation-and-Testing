#!/usr/bin/env python3
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
    
    # Get weekly data for pivot levels (primary filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day volume average
    vol_ma = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        for i in range(20, len(volume_1d)):
            vol_ma[i] = np.mean(volume_1d[i-20:i])
    
    # Align all data to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    vol_ma_6h = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-day average
        vol_confirm = volume[i] > 1.3 * vol_ma_6h[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation
            if close[i] > r3_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume confirmation
            elif close[i] < s3_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below R1 (take profit) or below S2 (stop)
            if close[i] < r1_6h[i]:
                signals[i] = 0.0  # exit long
                position = 0
            elif close[i] < s2_6h[i]:
                signals[i] = 0.0  # stop loss
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above S1 (take profit) or above R2 (stop)
            if close[i] > s1_6h[i]:
                signals[i] = 0.0  # exit short
                position = 0
            elif close[i] > r2_6h[i]:
                signals[i] = 0.0  # stop loss
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R3S3_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0