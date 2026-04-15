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
    
    # Weekly Donchian channels for trend direction (from weekly high/low)
    df_1w = get_htf_data(prices, '1w')
    donchian_high = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min()
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high.values)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low.values)
    
    # Weekly pivot points (from weekly close)
    weekly_close = df_1w['close'].values
    pivot = (df_1w['high'].values + df_1w['low'].values + weekly_close) / 3
    r1 = 2 * pivot - df_1w['low'].values
    s1 = 2 * pivot - df_1w['high'].values
    r2 = pivot + (df_1w['high'].values - df_1w['low'].values)
    s2 = pivot - (df_1w['high'].values - df_1w['low'].values)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long conditions:
        # 1. Price above weekly Donchian high (uptrend)
        # 2. Price above weekly pivot (bullish bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_aligned[i] and
            close[i] > pivot_aligned[i] and
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short conditions:
        # 1. Price below weekly Donchian low (downtrend)
        # 2. Price below weekly pivot (bearish bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_aligned[i] and
              close[i] < pivot_aligned[i] and
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit conditions:
        # Long exit: price crosses below weekly pivot
        elif i > 0 and signals[i-1] == 0.25 and close[i] < pivot_aligned[i]:
            signals[i] = 0.0
        
        # Short exit: price crosses above weekly pivot
        elif i > 0 and signals[i-1] == -0.25 and close[i] > pivot_aligned[i]:
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyDonchian_Pivot_Volume"
timeframe = "6h"
leverage = 1.0