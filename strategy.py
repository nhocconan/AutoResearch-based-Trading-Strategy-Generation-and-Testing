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
    
    # Get 12h data for Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) - vectorized
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    upper = np.full(len(high_12h), np.nan)
    lower = np.full(len(high_12h), np.nan)
    for i in range(20, len(high_12h)):
        upper[i] = np.max(high_12h[i-20:i])
        lower[i] = np.min(low_12h[i-20:i])
    donch_upper_12h = upper
    donch_lower_12h = lower
    donch_upper_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_upper_12h)
    donch_lower_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_lower_12h)
    
    # Get 1d data for weekly pivot (using daily high/low/close)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points from daily data
    # Using last 5 days to approximate weekly pivot (simplified)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot: (H + L + C)/3 for each day
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L, S1 = 2*P - H
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    # R2 = P + (H - L), S2 = P - (H - L)
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    # R4 = H + 3*(P - L), S4 = L - 3*(H - P)
    r4_1d = high_1d + 3 * (pivot_1d - low_1d)
    s4_1d = low_1d - 3 * (high_1d - pivot_1d)
    
    # Align to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Get 1d data for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian, volume MA, and pivot data
    start_idx = max(20, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_12h_aligned[i]) or np.isnan(donch_lower_12h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_12h_aligned[i]
        lower = donch_lower_12h_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Pivot levels
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 1d MA (volume breakout)
        vol_breakout = vol_now > 1.5 * vol_ma
        
        # Entry conditions
        if position == 0:
            # Long: break above R4 with volume (continuation)
            if close[i] > r4 and vol_breakout:
                signals[i] = size
                position = 1
            # Short: break below S4 with volume (continuation)
            elif close[i] < s4 and vol_breakout:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below R3 or volume drops
            if close[i] < r3 or vol_now < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above S3 or volume drops
            if close[i] > s3 or vol_now < 0.7 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dPivot_Volume"
timeframe = "6h"
leverage = 1.0