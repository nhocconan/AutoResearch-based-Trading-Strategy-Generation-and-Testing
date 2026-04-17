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
    
    # === 1w Weekly Pivot Points ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pivot = (high_1w + low_1w + close_1w) / 3
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    r2 = pivot + (high_1w - low_1w)
    s2 = pivot - (high_1w - low_1w)
    r3 = high_1w + 2 * (pivot - low_1w)
    s3 = low_1w - 2 * (high_1w - pivot)
    
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # === 6h Volume Spike (2.0x 20-period average) ===
    df_6h = get_htf_data(prices, '6h')
    volume_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    # === 1d ADX (14 periods) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation for 6h
    vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
    vol_confirm = vol_6h_aligned > vol_ma_20_6h_aligned * 2.0
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for calculations
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20_6h_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: ADX > 25 for trending market
        trending = adx_aligned[i] > 25
        
        # Entry logic: only enter when flat
        if position == 0 and trending:
            # Long: price breaks above R2 with volume confirmation
            if close[i] > r2_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S2 with volume confirmation
            elif close[i] < s2_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: price returns to opposite pivot level
        elif position == 1:
            # Exit long: price crosses below S1
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above R1
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R2_S2_Breakout_Volume_ADXFilter"
timeframe = "6h"
leverage = 1.0