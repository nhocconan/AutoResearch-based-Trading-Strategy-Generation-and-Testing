#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly high, low, close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly pivots to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3)
    
    # === 1d Volume Spike Detection ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Volume MA(20) - using pandas for efficiency
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume data to 6h
    volume_1d_6h = align_htf_to_ltf(prices, df_1d, volume_1d)
    vol_ma_20_6h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Volume spike: current volume > 2.0 x 20-period average
    volume_spike = volume_1d_6h > vol_ma_20_6h * 2.0
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(volume_1d_6h[i]) or np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R3 with volume spike
            if close[i] > r3_6h[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S3 with volume spike
            elif close[i] < s3_6h[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to pivot area
        elif position == 1:
            # Exit long: price returns below R1
            if close[i] < r1_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above S1
            if close[i] > s1_6h[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Breakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0