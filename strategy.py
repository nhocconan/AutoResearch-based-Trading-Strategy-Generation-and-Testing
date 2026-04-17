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
    
    # Get daily data for 1-day ATR and pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = 2 * pivot_1d - low_1d
    s1 = 2 * pivot_1d - high_1d
    r2 = pivot_1d + range_1d
    s2 = pivot_1d - range_1d
    r3 = high_1d + 2 * (pivot_1d - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot_1d)
    
    # Align daily pivots to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 6-day ATR for volatility filter (6d = 24 x 6h bars)
    # Use daily ATR scaled to 6h
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_1d[0] = high_low[0]
    atr_1d = pd.Series(tr_1d).rolling(window=6, min_periods=6).mean().values  # 6-day ATR
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # 6h close for entry/exit
    close_6h_series = pd.Series(close_6h)
    
    # Volume filter: 6h volume > 1.8x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (vol_ma_6h * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_6h[i]) or np.isnan(volume_filter_6h[i])):
            signals[i] = 0.0
            continue
        
        # Current 6h price
        price = close_6h[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume and above pivot
            if price > r3_aligned[i] and volume_filter_6h[i] and price > pivot_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume and below pivot
            elif price < s3_aligned[i] and volume_filter_6h[i] and price < pivot_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or volatility-based stop
            if price < s1_aligned[i] or price < (pivot_1d_aligned[i] - 1.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or volatility-based stop
            if price > r1_aligned[i] or price > (pivot_1d_aligned[i] + 1.5 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_Volume_ATRStop"
timeframe = "6h"
leverage = 1.0