#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla pivot calculation
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    
    # Resistance levels
    r4 = close_prev + range_val * 1.1 / 2
    r3 = close_prev + range_val * 1.1 / 4
    r2 = close_prev + range_val * 1.1 / 6
    r1 = close_prev + range_val * 1.1 / 12
    
    # Support levels
    s1 = close_prev - range_val * 1.1 / 12
    s2 = close_prev - range_val * 1.1 / 6
    s3 = close_prev - range_val * 1.1 / 4
    s4 = close_prev - range_val * 1.1 / 2
    
    # Align pivot levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d volume confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_10 = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_avg_10_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_10)
    
    # 4h ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure sufficient data
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_avg_10_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Current 1d volume (aligned)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_confirm = vol_1d_current > vol_avg_10_aligned[i]
        
        # Price levels
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        # Volatility filter: only trade when ATR > 10-period average
        atr_avg_10 = pd.Series(atr).rolling(window=10, min_periods=10).mean()[i]
        vol_filter = atr[i] > atr_avg_10
        
        # Long conditions: price touches S1 with volume and volatility confirmation
        long_signal = vol_confirm and vol_filter and (price <= s1_level * 1.002)
        
        # Short conditions: price touches R1 with volume and volatility confirmation
        short_signal = vol_confirm and vol_filter and (price >= r1_level * 0.998)
        
        # Exit conditions: price touches opposite level or pivot
        long_exit = price >= pivot_aligned[i] * 0.998
        short_exit = price <= pivot_aligned[i] * 1.002
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals