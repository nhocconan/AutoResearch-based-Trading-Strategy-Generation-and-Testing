#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v32"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data (HTF) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for the previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Calculate pivot and levels
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r4 = close_prev + range_prev * 1.1 / 2
    r3 = close_prev + range_prev * 1.1 / 4
    r2 = close_prev + range_prev * 1.1 / 6
    r1 = close_prev + range_prev * 1.1 / 12
    s1 = close_prev - range_prev * 1.1 / 12
    s2 = close_prev - range_prev * 1.1 / 6
    s3 = close_prev - range_prev * 1.1 / 4
    s4 = close_prev - range_prev * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter on 4h - 20-period average
    vol_series = pd.Series(prices['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = prices['volume'].values > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready (check for NaN in aligned arrays)
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        close_price = prices['close'].iloc[i]
        
        # Breakout signals with volume confirmation
        # Long: price breaks above R3 (strong resistance)
        long_signal = close_price > r3_aligned[i] and volume_ok[i]
        # Short: price breaks below S3 (strong support)
        short_signal = close_price < s3_aligned[i] and volume_ok[i]
        
        # Exit when price returns to mean (pivot level)
        exit_long = close_price < pivot[i]
        exit_short = close_price > pivot[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals