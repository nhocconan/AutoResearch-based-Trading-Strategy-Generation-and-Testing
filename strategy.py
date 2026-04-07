#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v2"
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
    
    # 1D HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (using previous day's data)
    r3 = pivot + (range_1d * 1.1 / 2)
    r2 = pivot + (range_1d * 1.1 / 4)
    r1 = pivot + (range_1d * 1.1 / 6)
    s1 = pivot - (range_1d * 1.1 / 6)
    s2 = pivot - (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 2)
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation (20-period average)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R2 or volume dries up
            if close[i] >= r2_aligned[i] or not volume_ok[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S2 or volume dries up
            if close[i] <= s2_aligned[i] or not volume_ok[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if not volume_ok[i]:
                signals[i] = 0.0
                continue
            
            # Long: price touches S1 with rejection (close > S1 and near S1)
            if close[i] > s1_aligned[i] and close[i] <= s1_aligned[i] * 1.002:
                position = 1
                signals[i] = 0.25
            # Short: price touches R1 with rejection (close < R1 and near R1)
            elif close[i] < r1_aligned[i] and close[i] >= r1_aligned[i] * 0.998:
                position = -1
                signals[i] = -0.25
    
    return signals