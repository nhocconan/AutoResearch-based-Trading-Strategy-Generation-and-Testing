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
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (based on prior week)
    # Using standard pivot point formula: P = (H + L + C)/3
    pivot_1w = np.full_like(close_1w, np.nan)
    r1_1w = np.full_like(close_1w, np.nan)
    s1_1w = np.full_like(close_1w, np.nan)
    r2_1w = np.full_like(close_1w, np.nan)
    s2_1w = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
        r1_1w[i] = 2 * pivot_1w[i] - low_1w[i-1]
        s1_1w[i] = 2 * pivot_1w[i] - high_1w[i-1]
        r2_1w[i] = pivot_1w[i] + (high_1w[i-1] - low_1w[i-1])
        s2_1w[i] = pivot_1w[i] - (high_1w[i-1] - low_1w[i-1])
    
    # Align pivot levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # Calculate 6-day ATR for volatility filter
    if len(high) < 6 or len(low) < 6 or len(close) < 6:
        return np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr6 = np.full_like(close, np.nan)
    for i in range(5, len(tr)):
        atr6[i] = np.mean(tr[i-5:i+1])
    
    # Calculate 6-day average volume
    vol_avg6 = np.full_like(volume, np.nan)
    for i in range(5, len(volume)):
        vol_avg6[i] = np.mean(volume[i-5:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or
            np.isnan(atr6[i]) or
            np.isnan(vol_avg6[i]) or
            vol_avg6[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume vs 6-period average
        volume_ratio = volume[i] / vol_avg6[i]
        
        # ATR filter: require sufficient volatility
        volatility_filter = atr6[i] > 0
        
        if position == 0:
            # Long: Price above weekly R1 + volume surge + volatility
            if (close[i] > r1_1w_aligned[i] and
                volume_ratio > 2.0 and
                volatility_filter):
                position = 1
                signals[i] = position_size
            # Short: Price below weekly S1 + volume surge + volatility
            elif (close[i] < s1_1w_aligned[i] and
                  volume_ratio > 2.0 and
                  volatility_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below weekly pivot
            if close[i] < pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above weekly pivot
            if close[i] > pivot_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_R1S1_Volume_Filter"
timeframe = "6h"
leverage = 1.0