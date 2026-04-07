#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_1d_Volume_Scalp
Hypothesis: On 12-hour timeframe, use daily Camarilla pivot levels (S3/S4 and R3/R4) with volume confirmation for mean reversion.
Long when price touches or crosses below S3/S4 with volume > 1.5x average, targeting the pivot point (P).
Short when price touches or crosses above R3/R4 with volume > 1.5x average, targeting the pivot point (P).
Exit when price reaches the pivot point (P) or reverses with strong volume.
Designed for 15-25 trades/year to minimize fee drag while capturing reversals at extreme levels.
Works in both bull/bear markets as Camarilla adapts to volatility and volume filter avoids false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_1d_Volume_Scalp"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: S1, S2, S3, S4 and R1, R2, R3, R4
    s4 = close_1d - (range_1d * 1.500)
    s3 = close_1d - (range_1d * 1.250)
    s2 = close_1d - (range_1d * 1.166)
    s1 = close_1d - (range_1d * 1.083)
    r1 = close_1d + (range_1d * 1.083)
    r2 = close_1d + (range_1d * 1.166)
    r3 = close_1d + (range_1d * 1.250)
    r4 = close_1d + (range_1d * 1.500)
    
    # Align to 12h timeframe (shifted by 1 day for non-look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Volume filter: 20-period average on 12h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches pivot point or reverses with strong volume
            if close[i] >= pivot_aligned[i] or (close[i] < close[i-1] and volume[i] > 2.0 * vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches pivot point or reverses with strong volume
            if close[i] <= pivot_aligned[i] or (close[i] > close[i-1] and volume[i] > 2.0 * vol_ma[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price touches or crosses below S3/S4
                if close[i] <= s3_aligned[i] or close[i] <= s4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price touches or crosses above R3/R4
                elif close[i] >= r3_aligned[i] or close[i] >= r4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals