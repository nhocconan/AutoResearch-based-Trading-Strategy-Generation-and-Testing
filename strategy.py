#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for pivot points and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily pivot points (classic formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    r2_1d = pivot_1d + (high_1d - low_1d)
    s2_1d = pivot_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pivot_1d)
    r4_1d = r3_1d + (high_1d - low_1d)
    s4_1d = s3_1d - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Daily volume for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: break above R1 with volume and weekly uptrend
            if (price > r1_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                price > ema_20_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and weekly downtrend
            elif (price < s1_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  price < ema_20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below pivot or volume drops
            if price < pivot_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above pivot or volume drops
            if price > pivot_aligned[i] or vol < 0.5 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1S1_Breakout_Volume_WeeklyTrend"
timeframe = "6h"
leverage = 1.0