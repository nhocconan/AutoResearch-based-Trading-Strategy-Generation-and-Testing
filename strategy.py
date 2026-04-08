#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)
    r2_1d = pivot_1d + (range_1d * 1.1 / 6)
    r3_1d = pivot_1d + (range_1d * 1.1 / 4)
    r4_1d = pivot_1d + (range_1d * 1.1 / 2)
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)
    s2_1d = pivot_1d - (range_1d * 1.1 / 6)
    s3_1d = pivot_1d - (range_1d * 1.1 / 4)
    s4_1d = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_12h = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_12h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1w trend: 21-period EMA
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(r2_12h[i]) or 
            np.isnan(r3_12h[i]) or np.isnan(r4_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(s2_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < S2 or trend fails
            if close[i] < s2_12h[i] or close[i] < ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > R2 or trend fails
            if close[i] > r2_12h[i] or close[i] > ema_21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter from 1w EMA
            bullish = close[i] > ema_21_1w_aligned[i]
            bearish = close[i] < ema_21_1w_aligned[i]
            
            # Long: price > R1 + bullish trend + volume
            if (close[i] > r1_12h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price < S1 + bearish trend + volume
            elif (close[i] < s1_12h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals