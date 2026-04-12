# 6h_1d_1w_camarilla_pivot_reversion_v1
# Fades at weekly R3/S3 and continues at weekly R4/S4 with volume confirmation
# Uses weekly pivots to capture both mean-reversion in range and breakout continuation in trend
# Works in bull/bear via volume filter and pivot-based structure
# Target: 50-150 total trades over 4 years (12-37/year)

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
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = r3_1w + (high_1w - low_1w)
    s4_1w = s3_1w - (high_1w - low_1w)
    
    # Calculate volume spike (daily)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_spike = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if vol_ma_20[i] > 0:
            vol_spike[i] = volume_1d[i] / vol_ma_20[i]
    
    # Align weekly pivots to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Align volume spike to 6h
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 6h ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = np.full(n, np.nan)
    for i in range(14, n):
        atr_6h[i] = np.mean(tr[i-14:i+1])
    
    atr_ma_10 = np.full(n, np.nan)
    for i in range(23, n):
        if not np.isnan(np.mean(atr_6h[i-9:i+1])):
            atr_ma_10[i] = np.mean(atr_6h[i-9:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s4_1w_aligned[i]) or np.isnan(vol_spike_aligned[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(atr_ma_10[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 50% of its MA to avoid low volatility
        vol_filter = atr_6h[i] > 0.5 * atr_ma_10[i]
        
        # Volume confirmation: volume spike > 1.5x average
        vol_confirm = vol_spike_aligned[i] > 1.5
        
        # Price levels
        price = close[i]
        
        # Fade at R3/S3 (mean reversion in range)
        fade_long = (price <= s3_1w_aligned[i] * 1.005) and vol_filter and vol_confirm
        fade_short = (price >= r3_1w_aligned[i] * 0.995) and vol_filter and vol_confirm
        
        # Breakout continuation at R4/S4 (trend following)
        breakout_long = (price >= r4_1w_aligned[i] * 0.995) and vol_filter and vol_confirm
        breakout_short = (price <= s4_1w_aligned[i] * 1.005) and vol_filter and vol_confirm
        
        # Exit conditions
        long_exit = (price >= pivot_1w_aligned[i]) or (position == 1 and price <= s3_1w_aligned[i] * 1.005)
        short_exit = (price <= pivot_1w_aligned[i]) or (position == -1 and price >= r3_1w_aligned[i] * 0.995)
        
        if (fade_long or breakout_long) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (fade_short or breakout_short) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_camarilla_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0