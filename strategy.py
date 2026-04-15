#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily high/low for pivot calculation (use previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Standard pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Align to 6h timeframe
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current > 1.3x median of last 24 bars (48h)
    vol_median = pd.Series(volume).rolling(window=24, min_periods=24).median()
    vol_threshold = 1.3 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(24, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long conditions: price breaks above R1 with volume, target R2
        if (close[i] > r1_6h[i] and close[i] <= r2_6h[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short conditions: price breaks below S1 with volume, target S2
        elif (close[i] < s1_6h[i] and close[i] >= s2_6h[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit conditions: price returns to pivot level
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] <= pivot_6h[i]) or
               (signals[i-1] == -0.25 and close[i] >= pivot_6h[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_Pivot_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0