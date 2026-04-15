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
    
    # Weekly close for weekly pivot calculation
    weekly = get_htf_data(prices, '1w')
    close_w = weekly['close'].values
    high_w = weekly['high'].values
    low_w = weekly['low'].values
    
    # Weekly pivot points (using previous week's data)
    pivot = np.full(len(close_w), np.nan)
    r1 = np.full(len(close_w), np.nan)
    s1 = np.full(len(close_w), np.nan)
    r2 = np.full(len(close_w), np.nan)
    s2 = np.full(len(close_w), np.nan)
    
    for i in range(1, len(close_w)):
        pivot[i] = (high_w[i-1] + low_w[i-1] + close_w[i-1]) / 3.0
        r1[i] = 2 * pivot[i] - low_w[i-1]
        s1[i] = 2 * pivot[i] - high_w[i-1]
        r2[i] = pivot[i] + (high_w[i-1] - low_w[i-1])
        s2[i] = pivot[i] - (high_w[i-1] - low_w[i-1])
    
    pivot_aligned = align_htf_to_ltf(prices, weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, weekly, s2)
    
    # 6h ATR(14) for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume threshold: 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above weekly R2 + volume spike + price above weekly pivot
        if (close[i] > r2_aligned[i] and 
            volume[i] > vol_threshold[i] and
            close[i] > pivot_aligned[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below weekly S2 + volume spike + price below weekly pivot
        elif (close[i] < s2_aligned[i] and 
              volume[i] > vol_threshold[i] and
              close[i] < pivot_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to weekly pivot
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < pivot_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > pivot_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_1w_Pivot_R2S2_Vol1.5x_Breakout"
timeframe = "6h"
leverage = 1.0