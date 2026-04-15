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
    
    # Weekly Donchian (20) - trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donch_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    
    # Daily pivot direction - bias
    df_1d = get_htf_data(prices, '1d')
    pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    r1 = 2 * pivot - df_1d['low'].values
    s1 = 2 * pivot - df_1d['high'].values
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current > 2.0x median of last 50 bars
    vol_median = pd.Series(volume).rolling(window=50, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price > weekly Donchian high AND price > daily pivot + volume
        if (close[i] > donch_high_20_aligned[i] and 
            close[i] > pivot_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price < weekly Donchian low AND price < daily pivot - volume
        elif (close[i] < donch_low_20_aligned[i] and 
              close[i] < pivot_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back to daily pivot (mean reversion to pivot)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < pivot_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > pivot_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyDonchian_Pivot_Volume"
timeframe = "6h"
leverage = 1.0